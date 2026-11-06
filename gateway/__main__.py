"""Command line: run the gateway, or deploy the cluster side.

    python -m gateway serve  --config config.toml
    python -m gateway deploy --config config.toml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shlex
import time
from pathlib import Path

from .cluster import ClusterError, SlurmBackend
from .config import ENGINES, Config, download_url, load

CLUSTER_FILES = ["sd_job.sh", "generate.py", "postprocess.py"]
SHARED_SIF = "/shared/container_repository/explorer/pytorch/pytorch_training.sif"
def token_counter(config: Config):
    """count(engine, text): prompt length in each engine's own tokenizer (Engine.tokenizer)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import AddedToken, Tokenizer

    tokenizers = {}
    for name, engine in ENGINES.items():
        repo, _, path = engine.tokenizer.partition(":")
        tokenizer = Tokenizer.from_file(hf_hub_download(repo, path))
        # A style's learned embedding tokens (e.g. <s0>) are one token each once the job loads them.
        extra = sorted({t for s in config.styles.values() if s.engine == name for t in s.embedding_tokens})
        if extra:
            tokenizer.add_tokens([AddedToken(t, single_word=False, normalized=False) for t in extra])
        tokenizers[name] = tokenizer
    return lambda engine, text: len(tokenizers[engine].encode(text).ids)


def serve(config: Config) -> None:
    import uvicorn

    from .app import create_app
    from .service import Service

    service = Service(config, SlurmBackend(config), token_counter(config), time.time)
    uvicorn.run(create_app(service), host=config.host, port=config.port)


async def deploy(config: Config) -> None:
    """Copy cluster/ to ~/<remote_dir>/<env>/ and pin the container (DESIGN.md 6.4)."""
    backend = SlurmBackend(config)
    root = config.root_dir
    env_dir = config.env_dir
    q = shlex.quote
    await backend._xfer(f"mkdir -p {q(env_dir)}/logs {q(env_dir)}/batches {q(env_dir)}/out")
    src = Path(__file__).resolve().parent.parent / "cluster"
    ssh = " ".join(q(a) for a in backend._ssh_args(config.xfer_host)[:-1])
    await backend._run(["rsync", "-a", "-e", ssh, *[str(src / f) for f in CLUSTER_FILES],
                        f"{config.user}@{config.xfer_host}:{env_dir}/"])
    print(f"cluster code -> ~/{env_dir}/")
    sif = f"{q(root)}/pytorch.sif"
    print(f"pinning container at ~/{root}/pytorch.sif (copies 4.3 GB the first time)...")
    await backend._xfer(f"test -f {sif} || {{ cp {q(SHARED_SIF)} {sif}.tmp && mv {sif}.tmp {sif}; }}",
                        timeout=3600)
    await deploy_styles(config, backend)
    print("deployed")


async def deploy_styles(config: Config, backend: SlurmBackend) -> None:
    """Download each style's files once into ~/<remote_dir>/loras/, checking SHA-256 (DESIGN.md 7.1).

    Downloads run on the transfer node. A Civitai token goes over ssh's stdin as a
    header, so it never appears on a command line on the cluster.
    """
    q = shlex.quote
    loras = f"{config.root_dir}/loras"
    for style in config.styles.values():
        files = [(style.source, style.sha256, style.lora_file)]
        if style.embeddings:
            files.append((style.embeddings, style.embeddings_sha256, style.embeddings_file))
        for source, sha, name in files:
            sha = sha.lower()
            target = q(f"{loras}/{name}")
            civitai = source.startswith("civitai:")
            if civitai and not config.civitai_token:
                raise SystemExit(f"style {style.name}: [downloads] civitai_token is needed for {source}")
            # curl -K - reads options from stdin. (-H @- would too, but the transfer
            # node's curl 7.29 silently ignores it and sends no header.)
            options = f'header = "Authorization: Bearer {config.civitai_token}"\n'.encode() if civitai else b""
            script = (
                f"if [ -f {target} ] && echo {sha}' '{target} | sha256sum -c --status; then echo present; exit 0; fi; "
                f"mkdir -p {q(loras)} && curl -fsSL --retry 3 {'-K -' if civitai else ''} -o {target}.tmp "
                f"{q(download_url(source))} && actual=$(sha256sum {target}.tmp | cut -c1-64) && "
                f'if [ "$actual" = {sha} ]; then mv {target}.tmp {target}; echo downloaded; '
                f'else rm -f {target}.tmp; echo "sha256 mismatch: got $actual" >&2; exit 3; fi'
            )
            try:
                result = await backend._xfer(script, stdin=options, timeout=3600)
            except ClusterError as exc:
                raise SystemExit(f"style {style.name}: {source}: {exc}") from None
            print(f"style {style.name}: {name} {result.strip()}")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m gateway", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["serve", "deploy"])
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load(args.config)
    if args.command == "serve":
        serve(config)
    else:
        asyncio.run(deploy(config))


if __name__ == "__main__":
    main()
