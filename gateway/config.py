"""Gateway settings, read from config.toml (DESIGN.md 9)."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

NAME = re.compile(r"[a-z0-9_]{1,32}")
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Engine:
    """An image model a task can run on (DESIGN.md 7)."""

    name: str
    gres: str  # Slurm GPU request; the model's memory needs decide it
    mem: str  # Slurm CPU memory: weights pass through it on their way to the GPU
    max_prompt_tokens: int  # what its text encoder reads; longer prompts are rejected
    tokenizer: str  # "<hf repo>:<path of tokenizer.json>", for counting prompt tokens
    guidance: str  # "always": negative prompt always used; "optional": only when the task gives one


ENGINES = {
    # SDXL: fits any GPU; its CLIP text encoders read 77 tokens; classic negative prompts.
    "sdxl": Engine("sdxl", "gpu:1", "32G", 77, "openai/clip-vit-large-patch14:tokenizer.json", "always"),
    # Qwen-Image-2.1: about 40 GB of GPU memory, so H200 (a100 mixes 40 and 80 GB cards).
    # Meant to be sampled without guidance; a negative prompt turns guidance on (twice the time).
    "qwen21": Engine("qwen21", "gpu:h200:1", "96G", 512, "Qwen/Qwen-Image-2.1:processor/tokenizer.json", "optional"),
}


@dataclass(frozen=True)
class Style:
    """A LoRA a task can ask for with "style": <name> (DESIGN.md 7.1)."""

    name: str
    source: str  # "hf:<owner>/<repo>/<file>" or "civitai:<model version id>"
    sha256: str
    trigger: str = ""  # added in front of the prompt
    scale: float = 1.0
    embeddings: str = ""  # optional textual-inversion file, same source forms
    embeddings_sha256: str = ""
    embedding_tokens: tuple[str, ...] = ()
    hint: str = ""  # how to prompt this style; shown on the test page
    engine: str = "sdxl"  # a LoRA works only with the model it was trained for

    def __post_init__(self) -> None:
        if not NAME.fullmatch(self.name):
            raise ValueError(f"style name must be 1-32 of a-z 0-9 _, got {self.name!r}")
        files = [(self.source, self.sha256)]
        if self.embeddings:
            files.append((self.embeddings, self.embeddings_sha256))
        for src, sha in files:
            download_url(src)  # raises on a bad source
            if not SHA256.fullmatch(sha.lower()):
                raise ValueError(f"style {self.name}: sha256 of {src} must be 64 hex digits")
        if self.embeddings and not self.embedding_tokens:
            raise ValueError(f"style {self.name}: embeddings need embedding_tokens")
        if self.engine not in ENGINES:
            raise ValueError(f"style {self.name}: engine must be one of {sorted(ENGINES)}")
        if not 0 < self.scale <= 2:
            raise ValueError(f"style {self.name}: scale must be in (0, 2]")

    @property
    def lora_file(self) -> str:
        """File name under <remote_dir>/loras/; the hash in it makes a new file a new name."""
        return f"{self.name}-{self.sha256.lower()[:12]}.safetensors"

    @property
    def embeddings_file(self) -> str:
        return f"{self.name}-{self.embeddings_sha256.lower()[:12]}.emb.safetensors" if self.embeddings else ""

    def full_prompt(self, prompt: str) -> str:
        """The prompt the model reads: the trigger in front, unless the client already placed it."""
        if not self.trigger or self.trigger.lower() in prompt.lower():
            return prompt
        return f"{self.trigger}, {prompt}"


def download_url(source: str) -> str:
    kind, _, rest = source.partition(":")
    if kind == "hf" and re.fullmatch(r"[\w.-]+/[\w.-]+/[\w./-]+\.safetensors", rest):
        owner, repo, path = rest.split("/", 2)
        return f"https://huggingface.co/{owner}/{repo}/resolve/main/{path}"
    if kind == "civitai" and rest.isdigit():
        return f"https://civitai.com/api/download/models/{rest}"
    raise ValueError(f"source must be hf:<owner>/<repo>/<file>.safetensors or civitai:<version id>, got {source!r}")


@dataclass(frozen=True)
class Config:
    # [server]
    host: str = "127.0.0.1"
    port: int = 8080
    data_dir: Path = Path("data")
    # [cluster]
    user: str = ""
    login_host: str = "login.explorer.northeastern.edu"
    xfer_host: str = "xfer.discovery.neu.edu"
    ssh_key: str = ""  # empty: ssh's own default key
    remote_dir: str = "sd_service"  # relative to the cluster home directory
    env: str = "dev"
    # [queue]
    max_batch_images: int = 200
    max_wait_s: float = 60
    max_attempts: int = 3
    batcher_interval_s: float = 5
    idle_s: float = 300  # how long a job stays warm after its batch; 0 = exit at once
    handoff_timeout_s: float = 90  # take back a hand-off no worker claimed by then
    warm_wait_s: float = 5  # max_wait_s while a job is warm: no cold start to gather tasks for
    job_minutes: int = 90  # Slurm time limit: this plus 15 s per image of the first batch
    # [poll]
    poll_interval_s: float = 30
    unreachable_after_s: float = 300
    resubmit_after_s: float = 300
    # [limits]
    max_tasks_per_request: int = 100
    max_ids_per_status: int = 100
    min_disk_free_gb: float = 10
    # [generation]
    default_engine: str = "qwen21"  # for tasks that name neither an engine nor a style
    # [downloads]
    civitai_token: str = ""  # only `deploy` uses it, to download Civitai LoRAs
    # [[styles]]
    styles: dict[str, Style] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9]{1,16}", self.env):
            raise ValueError(f"env must be 1-16 lowercase letters or digits, got {self.env!r}")
        if self.default_engine not in ENGINES:
            raise ValueError(f"default_engine must be one of {sorted(ENGINES)}, got {self.default_engine!r}")

    @property
    def root_dir(self) -> str:
        """Shared by all environments: container, model cache, loras/. Relative to home."""
        return self.remote_dir.removeprefix("~/").rstrip("/")

    @property
    def env_dir(self) -> str:
        """This environment's directory on the cluster, relative to the home directory."""
        return f"{self.root_dir}/{self.env}"

    @property
    def batch_prefix(self) -> str:
        """Batch IDs, which are also Slurm job names: sdg-<env>-000001."""
        return f"sdg-{self.env}-"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "gateway.db"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"


_SECTIONS = {
    "server": {"listen", "data_dir"},
    "cluster": {"user", "login_host", "xfer_host", "ssh_key", "remote_dir", "env"},
    "queue": {"max_batch_images", "max_wait_s", "max_attempts", "batcher_interval_s", "idle_s", "handoff_timeout_s",
              "job_minutes", "warm_wait_s"},
    "poll": {"interval_s", "unreachable_after_s", "resubmit_after_s"},
    "limits": {"max_tasks_per_request", "max_ids_per_status", "min_disk_free_gb"},
    "generation": {"default_engine"},
    "downloads": {"civitai_token"},
}
_STYLE_KEYS = {"name", "source", "sha256", "trigger", "scale", "embeddings", "embeddings_sha256", "embedding_tokens",
               "hint", "engine"}


def load(path: Path) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    values: dict = {}
    styles = raw.pop("styles", [])
    if not isinstance(styles, list):
        raise ValueError(f"{path}: styles must be [[styles]] tables")
    values["styles"] = {}
    for entry in styles:
        unknown = set(entry) - _STYLE_KEYS
        if unknown:
            raise ValueError(f"{path}: unknown key(s) {sorted(unknown)} in [[styles]]")
        entry = {**entry, "embedding_tokens": tuple(entry.get("embedding_tokens", ()))}
        style = Style(**entry)
        if style.name in values["styles"]:
            raise ValueError(f"{path}: style {style.name!r} is defined twice")
        values["styles"][style.name] = style
    for section, keys in raw.items():
        if section not in _SECTIONS:
            raise ValueError(f"{path}: unknown section [{section}]")
        for key, value in keys.items():
            if key not in _SECTIONS[section]:
                raise ValueError(f"{path}: unknown key {key!r} in [{section}]")
            if key == "listen":
                host, _, port = value.rpartition(":")
                values["host"], values["port"] = host, int(port)
            elif key == "data_dir":
                values["data_dir"] = (path.parent / value).resolve()
            elif section == "poll" and key == "interval_s":
                values["poll_interval_s"] = value
            else:
                values[key] = value
    return Config(**values)
