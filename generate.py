#!/usr/bin/env python3
"""Generate kids' dictionary illustrations with SDXL from a CSV of words.

CSV columns: word (required), description (optional, e.g. "a red apple").
Writes <out>/<word>_<n>.png and appends one line per image to <out>/metadata.jsonl.
Images that already exist are skipped, so a re-run continues where it stopped.

Each image gets a pure white background (see postprocess.py). An image that looks
like a repeating pattern instead of one subject is regenerated with a new seed, up
to MAX_TRIES times; if every try fails it is kept and marked needs_review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

from postprocess import looks_like_pattern, whiten_background

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
# SDXL's own VAE overflows in fp16 and can return black images; this is the standard fix.
VAE = "madebyollin/sdxl-vae-fp16-fix"
MAX_TRIES = 4

# "Sticker" anchors SDXL on one isolated object on white. Words like "dictionary" or
# "picture book" pull in book props and pages full of small repeated objects.
STYLE = (
    "a cute cartoon sticker of {subject}, single object, isolated on white background, "
    "simple flat colors, soft rounded shapes, clean bold outlines, centered, for young children"
)
NEGATIVE = (
    "pattern, repeating, seamless, wallpaper, collage, many objects, duplicate, book, "
    "colored background, scenery, text, letters, watermark, signature, "
    "scary, creepy, violent, blood, weapon, nsfw, nude, photo, photorealistic, dark, "
    "deformed, extra limbs, blurry"
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def seed_for(word: str, n: int, attempt: int = 0) -> int:
    """Stable per-image seed, so the same word always gives the same picture."""
    key = f"{word}:{n}" if attempt == 0 else f"{word}:{n}:{attempt}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def read_words(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("word") or "").strip()]
    return [{"word": r["word"].strip(), "description": (r.get("description") or "").strip()} for r in rows]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="CSV with word[,description]")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variants", type=int, default=2, help="images per word")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--size", type=int, default=1024)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    words = read_words(args.input)
    print(f"{len(words)} words x {args.variants} variants from {args.input}", flush=True)

    vae = AutoencoderKL.from_pretrained(VAE, dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, vae=vae, dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def generate(prompt: str, seeds: list[int]):
        return pipe(
            prompt=[prompt] * len(seeds),
            negative_prompt=[NEGATIVE] * len(seeds),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.size,
            width=args.size,
            generator=[torch.Generator("cuda").manual_seed(s) for s in seeds],
        ).images

    done = skipped = flagged = 0
    with open(args.out / "metadata.jsonl", "a", encoding="utf-8") as meta:
        for row in words:
            word, subject = row["word"], row["description"] or row["word"]
            todo = [n for n in range(1, args.variants + 1)
                    if not (args.out / f"{slug(word)}_{n}.png").exists()]
            skipped += args.variants - len(todo)
            if not todo:
                continue
            prompt = STYLE.format(subject=subject)
            t = time.monotonic()
            # First try for all variants in one batch; retries one image at a time.
            first = generate(prompt, [seed_for(word, n) for n in todo])
            for n, image in zip(todo, first):
                attempt = 0
                while True:
                    image, stats = whiten_background(image)
                    if not looks_like_pattern(stats) or attempt + 1 >= MAX_TRIES:
                        break
                    attempt += 1
                    image = generate(prompt, [seed_for(word, n, attempt)])[0]
                review = looks_like_pattern(stats)
                flagged += review
                name = f"{slug(word)}_{n}.png"
                image.save(args.out / name)
                meta.write(json.dumps({"file": name, "word": word, "prompt": prompt,
                                       "negative_prompt": NEGATIVE, "seed": seed_for(word, n, attempt),
                                       "attempts": attempt + 1, "needs_review": review, **stats,
                                       "steps": args.steps, "guidance": args.guidance,
                                       "size": args.size, "model": MODEL}) + "\n")
                meta.flush()
                if attempt or review:
                    print(f"  {name}: {attempt + 1} tries{', NEEDS REVIEW' if review else ''}", flush=True)
            done += len(todo)
            print(f"{word}: {len(todo)} image(s) in {time.monotonic() - t:.1f}s", flush=True)

    print(f"generated {done} ({flagged} need review), skipped {skipped} existing, output in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
