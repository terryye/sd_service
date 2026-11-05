# Kids' dictionary illustration service

Generates sticker-style illustrations (one subject, pure white background) with SDXL on Explorer.

## Use

1. Write a CSV with a `word` column and an optional `description` column (see `words.csv`):

   ```csv
   word,description
   apple,a shiny red apple with a green leaf
   cat,a small orange cat sitting
   ```

2. From this folder on your Mac, run:

   ```bash
   ./submit.sh mywords.csv                  # 2 images per word
   ./submit.sh mywords.csv --variants 4     # more choices per word
   ```

   This uploads the CSV, runs one GPU job, waits for it, and downloads the images to `images/mywords/`.

Each word takes about 17 s per 2 images. Images that already exist are skipped, so you can re-run an interrupted batch with the same command. To regenerate a word, delete its files in `~/sd_service/out/<batch>/` on the cluster.

`metadata.jsonl` records the prompt, seed and checks for every image. Images marked `"needs_review": true` still looked like a pattern after 4 tries.

## Files

| File | Where it runs | What it does |
|---|---|---|
| `submit.sh` | Mac | uploads, submits, waits, downloads |
| `sd_job.sh` | cluster (Slurm) | GPU job: installs `diffusers` on node `/tmp`, runs `generate.py` |
| `generate.py` | cluster | SDXL generation, retries patterned images |
| `postprocess.py` | cluster | whitens the background, detects patterns |

On the cluster, everything lives in `~/sd_service`: code, the SDXL model (`hf_cache/`, ~7 GB), outputs (`out/`) and logs (`logs/`).
