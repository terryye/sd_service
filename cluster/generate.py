#!/usr/bin/env python3
"""The image worker of one Slurm job (DESIGN.md 6.1, 6.5 and 7).

Runs the batch it was started for, then stays warm for up to --idle-s seconds, in
case the gateway hands it another batch, and runs that too. The model (the engine
named in the first batch: SDXL or Qwen-Image-2.1) and any LoRA styles are loaded
once per job; a warm job only takes batches for its own engine.

For each batch, under the environment directory:
  reads  batches/<batch_id>/batch.json
  writes out/<batch_id>/<taskid>/v<version>/<prefix>_<n>.webp, one line per task in
         out/<batch_id>/progress.jsonl, and out/<batch_id>/finished at the end.
A re-run skips tasks that already have a progress line.

While warm, it announces itself in workers/<job_id>.json and watches inbox/<job_id>,
a file the gateway writes holding the next batch ID. It claims a hand-off by renaming
that file (so the gateway can take back an unclaimed one) and marks it with
out/<batch_id>/claimed.

postprocess "sticker": the background becomes pure white (see postprocess.py), and
an image that looks like a repeating pattern, or has an object cut off by the edge,
is regenerated with the next seed, up to MAX_TRIES times. If every try fails, the
model's last image is kept as drawn (not whitened) and marked needs_review.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import diffusers
import torch
from postprocess import needs_retry, whiten_background

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
# SDXL's own VAE overflows in fp16 and can return black images; this is the standard fix.
SDXL_VAE = "madebyollin/sdxl-vae-fp16-fix"
SDXL_STEPS = 30
SDXL_GUIDANCE = 6.0
QWEN21 = "Qwen/Qwen-Image-2.1"
QWEN21_STEPS = 40  # the model card's setting
QWEN21_CFG = 4.0  # only used when a task has a negative prompt; 2.1 is meant to run without guidance
SIZE = 1024
MAX_TRIES = 4
WEBP_QUALITY = 90
INBOX_POLL_S = 2
MIN_JOB_LEFT_S = 600  # stop waiting for work this long before the Slurm time limit


def retry_seed(seed: int, attempt: int) -> int:
    return (seed + attempt * 0x9E3779B1) % 2**32


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp(msg: str) -> None:
    """A wall-clock line like sd_job.sh's, so the job log shows where the time goes."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def done_tasks(progress: Path) -> set[tuple[str, int]]:
    if not progress.exists():
        return set()
    done = set()
    for line in progress.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by a crash; the task runs again
        done.add((rec["taskid"], rec["version"]))
    return done


def append_line(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_webp(image, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    image.save(tmp, "WEBP", quality=WEBP_QUALITY, method=6)
    os.replace(tmp, path)


LORA_PARTS = ("unet", "text_encoder", "text_encoder_2")


def fix_prefixes(state_dict: dict) -> dict:
    """Collapse doubled part prefixes, e.g. "unet.unet.down_blocks..." to "unet.down_blocks...".

    Some older SDXL LoRA training scripts saved keys that way; diffusers strips one
    "unet." and then finds no matching layers.
    """
    fixed = {}
    for key, value in state_dict.items():
        for part in LORA_PARTS:
            if key.startswith(f"{part}.{part}."):
                key = key[len(part) + 1:]
                break
        fixed[key] = value
    return fixed


def load_style(pipe, adapter: str, style: dict, loras: Path) -> None:
    """A LoRA, plus its learned embedding tokens if it has any (pivotal tuning).

    Embeddings are expected in the diffusers SDXL format: a safetensors file with
    "clip_l" and "clip_g" tensors, one per text encoder.
    """
    from safetensors.torch import load_file

    pipe.load_lora_weights(fix_prefixes(load_file(str(loras / style["lora"]))), adapter_name=adapter)
    if style.get("embeddings"):
        emb = load_file(str(loras / style["embeddings"]))
        pipe.load_textual_inversion(emb["clip_l"], token=style["tokens"],
                                    text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
        pipe.load_textual_inversion(emb["clip_g"], token=style["tokens"],
                                    text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)


def adapter_name(style: dict) -> str:
    """The LoRA file name is unique per file contents, so a changed style loads as a new adapter."""
    return Path(style["lora"]).stem.replace(".", "_")


def load_pipeline(engine: str):
    if engine == "qwen21":
        pipe = diffusers.QwenImage21Pipeline.from_pretrained(QWEN21, dtype=torch.bfloat16)
    elif engine == "sdxl":
        vae = diffusers.AutoencoderKL.from_pretrained(SDXL_VAE, dtype=torch.float16)
        stamp("VAE loaded")
        pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
            SDXL, vae=vae, dtype=torch.float16, variant="fp16", use_safetensors=True
        )
    else:
        raise ValueError(f"unknown engine {engine!r}")
    stamp(f"{engine} pipeline loaded")
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    stamp("moved to GPU")
    return pipe


class Worker:
    def __init__(self, pipe, engine: str, env: Path, loras: Path):
        self.pipe = pipe
        self.engine = engine
        self.env = env
        self.loras = loras
        self.loaded: set[str] = set()  # adapters loaded in this job
        self.broken: dict[str, str] = {}  # adapter -> load error; fails only its own tasks

    def generate(self, prompt: str, negative: str, seeds: list[int]):
        n = len(seeds)
        common = dict(prompt=[prompt] * n, height=SIZE, width=SIZE,
                      generator=[torch.Generator("cuda").manual_seed(s) for s in seeds])
        if self.engine == "qwen21":
            if negative:  # the gateway sends one only when the task asked for it
                common.update(negative_prompt=[negative] * n, true_cfg_scale=QWEN21_CFG)
            return self.pipe(num_inference_steps=QWEN21_STEPS, **common).images
        return self.pipe(negative_prompt=[negative] * n, num_inference_steps=SDXL_STEPS,
                         guidance_scale=SDXL_GUIDANCE, **common).images

    def run_batch(self, batch_id: str) -> None:
        batch = json.loads((self.env / "batches" / batch_id / "batch.json").read_text(encoding="utf-8"))
        out = self.env / "out" / batch_id
        out.mkdir(parents=True, exist_ok=True)
        progress = out / "progress.jsonl"
        done = done_tasks(progress)
        todo = [t for t in batch["tasks"] if (t["taskid"], t["version"]) not in done]
        stamp(f"{batch_id}: {len(batch['tasks'])} tasks, {len(todo)} to do")
        engine = batch.get("engine", "sdxl")
        if engine != self.engine:  # the gateway only hands a warm job its own engine; refuse, don't guess
            for task in todo:
                append_line(progress, {"taskid": task["taskid"], "version": task["version"], "ts": now(),
                                       "error": f"batch for engine {engine} reached a {self.engine} job"})
            write_atomic(out / "finished", now())
            return

        # Load the styles this batch needs that this job hasn't loaded yet.
        styles = batch.get("styles", {})
        for name in sorted({t["style"] for t in todo if t.get("style")}):
            adapter = adapter_name(styles[name])
            if adapter in self.loaded or adapter in self.broken:
                continue
            try:
                load_style(self.pipe, adapter, styles[name], self.loras)
                self.loaded.add(adapter)
                stamp(f"style {name} loaded")
            except Exception as exc:
                self.broken[adapter] = f"style {name} failed to load: {type(exc).__name__}: {exc}"[:500]
                stamp(self.broken[adapter][:200])

        for task in todo:
            t = time.monotonic()
            adapter = adapter_name(styles[task["style"]]) if task.get("style") else None
            if adapter in self.broken:
                append_line(progress, {"taskid": task["taskid"], "version": task["version"],
                                       "error": self.broken[adapter], "ts": now()})
                continue
            if self.loaded:
                if adapter:
                    self.pipe.enable_lora()
                    self.pipe.set_adapters([adapter], adapter_weights=[styles[task["style"]]["scale"]])
                else:
                    self.pipe.disable_lora()
            record = self.run_task(task, out)
            record["ts"] = now()
            append_line(progress, record)
            stamp(f"{task['taskid']}: {len(record.get('images', []))} image(s) in "
                  f"{time.monotonic() - t:.1f}s{'  ERROR ' + record['error'] if 'error' in record else ''}")
        write_atomic(out / "finished", now())
        stamp(f"{batch_id}: finished")

    def run_task(self, task: dict, out: Path) -> dict:
        sticker = task["postprocess"] == "sticker"
        negative = task["negative_prompt"]  # built by the gateway: safety terms, sticker terms, the client's own
        task_dir = out / task["taskid"] / f"v{task['version']}"
        task_dir.mkdir(parents=True, exist_ok=True)
        record = {"taskid": task["taskid"], "version": task["version"], "images": []}
        try:
            # All candidates in one batched pass; pattern retries one image at a time.
            first = self.generate(task["prompt"], negative, task["seeds"])
            for n, (seed, image) in enumerate(zip(task["seeds"], first, strict=True), start=1):
                attempt, used = 0, seed
                result = "success"
                if sticker:
                    while True:
                        drawn = image
                        image, stats = whiten_background(drawn)
                        if not needs_retry(stats) or attempt + 1 >= MAX_TRIES:
                            break
                        attempt += 1
                        used = retry_seed(seed, attempt)
                        image = self.generate(task["prompt"], negative, [used])[0]
                    if needs_retry(stats):
                        # Not a sticker after all (often a full scene): whitening would only damage
                        # it, so the reviewer gets what the model drew.
                        image, result = drawn, "needs_review"
                name = f"{task['filename_prefix']}_{n}.webp"
                save_webp(image, task_dir / name)
                record["images"].append({"file": name, "seed": used, "result": result, "attempts": attempt + 1})
        except Exception as exc:  # one bad task must not stop the batch
            record = {"taskid": task["taskid"], "version": task["version"],
                      "error": f"{type(exc).__name__}: {exc}"[:500]}
        return record

    def wait_for_handoff(self, job_id: str, idle_s: float, job_end: float) -> str | None:
        """Stay warm until the gateway hands over a batch, or the idle window closes."""
        deadline = min(time.time() + idle_s, job_end - MIN_JOB_LEFT_S)
        if idle_s <= 0 or deadline <= time.time():
            return None
        workers, inbox = self.env / "workers", self.env / "inbox"
        workers.mkdir(exist_ok=True)
        inbox.mkdir(exist_ok=True)
        announce = workers / f"{job_id}.json"
        write_atomic(announce, json.dumps({"job_id": job_id, "idle_until": deadline, "job_end": job_end,
                                           "engine": self.engine}))
        stamp(f"warm: waiting up to {deadline - time.time():.0f}s for another batch")
        try:
            box, claimed = inbox / job_id, inbox / f"{job_id}.claimed"
            poke = inbox / f".poke-{job_id}"
            while time.time() < deadline:
                # Shared storage is NFS: this node caches that the inbox file is missing for up to
                # ~30 s. A write of its own makes it refetch the directory, so a hand-off shows at once.
                poke.touch()
                poke.unlink(missing_ok=True)
                try:
                    os.rename(box, claimed)  # atomic: the gateway can no longer take it back
                except FileNotFoundError:
                    time.sleep(INBOX_POLL_S)
                    continue
                batch_id = claimed.read_text(encoding="utf-8").strip()
                claimed.unlink()
                (self.env / "out" / batch_id).mkdir(parents=True, exist_ok=True)
                write_atomic(self.env / "out" / batch_id / "claimed", now())
                stamp(f"handed {batch_id}")
                return batch_id
            stamp("idle window closed")
            return None
        finally:
            announce.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", type=Path, required=True, help="the environment directory, ~/sd_service/<env>")
    p.add_argument("--batch-id", required=True, help="the batch this job was submitted for")
    p.add_argument("--loras", type=Path, required=True, help="directory holding the styles' LoRA files")
    p.add_argument("--job-id", default="local")
    p.add_argument("--job-end", type=float, default=0, help="Slurm end time of this job, Unix seconds")
    p.add_argument("--idle-s", type=float, default=0, help="how long to stay warm after a batch; 0 = exit")
    args = p.parse_args()

    stamp("imports done")
    first = json.loads((args.env / "batches" / args.batch_id / "batch.json").read_text(encoding="utf-8"))
    engine = first.get("engine", "sdxl")
    worker = Worker(load_pipeline(engine), engine, args.env, args.loras)
    batch_id = args.batch_id
    while batch_id:
        worker.run_batch(batch_id)
        batch_id = worker.wait_for_handoff(args.job_id, args.idle_s, args.job_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
