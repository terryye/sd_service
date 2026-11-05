#!/usr/bin/env python3
"""Generate one gateway batch with SDXL (DESIGN.md 6.1 and 7).

Reads  <env>/batches/<batch_id>/batch.json
Writes <env>/out/<batch_id>/<taskid>/v<version>/<prefix>_<n>.webp and, after each
task, one line in <env>/out/<batch_id>/progress.jsonl. The gateway collects a task
as soon as its line appears. A re-run skips tasks that already have a line.

postprocess "sticker": the background becomes pure white (see postprocess.py), and
an image that looks like a repeating pattern, or has an object cut off by the edge,
is regenerated with the next seed, up to MAX_TRIES times. If every try fails, the image is kept
and marked needs_review.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from postprocess import needs_retry, whiten_background

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
# SDXL's own VAE overflows in fp16 and can return black images; this is the standard fix.
VAE = "madebyollin/sdxl-vae-fp16-fix"
SIZE = 1024
STEPS = 30
GUIDANCE = 6.0
MAX_TRIES = 4
WEBP_QUALITY = 90

# Always added: keeps every image suitable for young children.
SAFETY_NEGATIVE = (
    "scary, creepy, violent, blood, weapon, nsfw, nude, dark, deformed, extra limbs, "
    "text, letters, watermark, signature"
)
# Added for "sticker" tasks: pushes SDXL toward one object on a plain background.
STICKER_NEGATIVE = (
    "pattern, repeating, seamless, wallpaper, collage, many objects, duplicate, "
    "colored background, scenery, photo, photorealistic, blurry"
)


def retry_seed(seed: int, attempt: int) -> int:
    return (seed + attempt * 0x9E3779B1) % 2**32


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp(msg: str) -> None:
    """A wall-clock line like sd_job.sh's, so the job log shows where start-up time goes."""
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


def load_style(pipe, name: str, style: dict, loras: Path) -> None:
    """A LoRA, plus its learned embedding tokens if it has any (pivotal tuning).

    Embeddings are expected in the diffusers SDXL format: a safetensors file with
    "clip_l" and "clip_g" tensors, one per text encoder.
    """
    from safetensors.torch import load_file

    pipe.load_lora_weights(fix_prefixes(load_file(str(loras / style["lora"]))), adapter_name=name)
    if style.get("embeddings"):
        emb = load_file(str(loras / style["embeddings"]))
        pipe.load_textual_inversion(emb["clip_l"], token=style["tokens"],
                                    text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
        pipe.load_textual_inversion(emb["clip_g"], token=style["tokens"],
                                    text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=Path, required=True, help="path to batch.json")
    p.add_argument("--out", type=Path, required=True, help="out/<batch_id> directory")
    p.add_argument("--loras", type=Path, required=True, help="directory holding the styles' LoRA files")
    args = p.parse_args()

    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    progress = args.out / "progress.jsonl"
    done = done_tasks(progress)
    todo = [t for t in batch["tasks"] if (t["taskid"], t["version"]) not in done]
    stamp(f"imports done; {batch['batch_id']}: {len(batch['tasks'])} tasks, {len(todo)} to do")
    if not todo:
        return 0

    vae = AutoencoderKL.from_pretrained(VAE, dtype=torch.float16)
    stamp("VAE loaded")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, vae=vae, dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    stamp("pipeline loaded")
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    stamp("moved to GPU")

    # Load only the styles this batch's remaining tasks use, once; tasks switch between them.
    styles = batch.get("styles", {})
    in_use = sorted({t["style"] for t in todo if t.get("style")})
    broken: dict[str, str] = {}  # a style that fails to load fails only its own tasks
    for name in in_use:
        try:
            load_style(pipe, name, styles[name], args.loras)
            stamp(f"style {name} loaded")
        except Exception as exc:
            broken[name] = f"style {name} failed to load: {type(exc).__name__}: {exc}"[:500]
            stamp(broken[name][:200])
    in_use = [n for n in in_use if n not in broken]

    def generate(prompt: str, negative: str, seeds: list[int]):
        return pipe(
            prompt=[prompt] * len(seeds),
            negative_prompt=[negative] * len(seeds),
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            height=SIZE,
            width=SIZE,
            generator=[torch.Generator("cuda").manual_seed(s) for s in seeds],
        ).images

    for task in todo:
        t = time.monotonic()
        if task.get("style") in broken:
            append_line(progress, {"taskid": task["taskid"], "version": task["version"],
                                   "error": broken[task["style"]], "ts": now()})
            continue
        if in_use:
            if task.get("style"):
                pipe.enable_lora()
                pipe.set_adapters([task["style"]], adapter_weights=[styles[task["style"]]["scale"]])
            else:
                pipe.disable_lora()
        sticker = task["postprocess"] == "sticker"
        negative = f"{SAFETY_NEGATIVE}, {STICKER_NEGATIVE}" if sticker else SAFETY_NEGATIVE
        task_dir = args.out / task["taskid"] / f"v{task['version']}"
        task_dir.mkdir(parents=True, exist_ok=True)
        record = {"taskid": task["taskid"], "version": task["version"], "images": []}
        try:
            # All candidates in one batched pass; pattern retries one image at a time.
            first = generate(task["prompt"], negative, task["seeds"])
            for n, (seed, image) in enumerate(zip(task["seeds"], first, strict=True), start=1):
                attempt, used = 0, seed
                result = "success"
                if sticker:
                    while True:
                        image, stats = whiten_background(image)
                        if not needs_retry(stats) or attempt + 1 >= MAX_TRIES:
                            break
                        attempt += 1
                        used = retry_seed(seed, attempt)
                        image = generate(task["prompt"], negative, [used])[0]
                    if needs_retry(stats):
                        result = "needs_review"
                name = f"{task['filename_prefix']}_{n}.webp"
                save_webp(image, task_dir / name)
                record["images"].append({"file": name, "seed": used, "result": result,
                                         "attempts": attempt + 1})
        except Exception as exc:  # one bad task must not stop the batch
            record = {"taskid": task["taskid"], "version": task["version"],
                      "error": f"{type(exc).__name__}: {exc}"[:500]}
        record["ts"] = now()
        append_line(progress, record)
        stamp(f"{task['taskid']}: {len(record.get('images', []))} image(s) in "
              f"{time.monotonic() - t:.1f}s{'  ERROR ' + record['error'] if 'error' in record else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
