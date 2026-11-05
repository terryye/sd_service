# sd_service

An internal image generation service for a kids' dictionary app. The app's backend submits prompts over HTTP; the service generates sticker-style illustrations (one subject, pure white background, 1024×1024 WebP) with SDXL on GPUs of Northeastern's Explorer HPC cluster, and serves the results.

## Status

| Part | State |
|---|---|
| Image pipeline on Explorer (`generate.py`, `postprocess.py`, `sd_job.sh`) | working, tested on the cluster |
| HTTP gateway (API, SQLite queue, batcher, poller, test page) | designed in [DESIGN.md](DESIGN.md), not built yet |

## How it works

```
app backend ──HTTP──> gateway ──SSH, rsync──> Explorer: Slurm GPU job (SDXL)
```

The gateway runs outside the cluster: it queues tasks in SQLite, batches them into Slurm GPU jobs (up to 200 images, or after 60 s), polls the job every 30 s, and serves the finished images. All connections start at the gateway; the cluster never calls back.

## API (planned)

| Endpoint | Purpose |
|---|---|
| `POST /api/images/gen` | submit up to 100 tasks; each is queued or rejected |
| `GET /api/images/tasks/{taskid}` | status and image URLs of one task |
| `GET /api/images/tasks?ids=a,b,c` | bulk status |
| `GET /api/health` | cluster reachability, queue length, free disk |
| `GET /results/{taskid}/{prefix}_{n}.webp` | an image |
| `GET /` | test page: submit a prompt, watch the status, preview the images |

A task is one prompt with 1 to 4 candidate images:

```json
{"taskid": "apple-01", "prompt": "a shiny red apple, cartoon sticker",
 "filename_prefix": "apple", "count": 4, "on_duplicate": "reject"}
```

See [DESIGN.md](DESIGN.md) for the full design.

## Files

| File | Runs on | What it does |
|---|---|---|
| `generate.py` | cluster | SDXL generation; retries images that come out as repeating patterns |
| `postprocess.py` | cluster | turns the background white, detects patterns |
| `sd_job.sh` | cluster (Slurm) | GPU job: installs `diffusers` on node-local `/tmp`, runs `generate.py` |
| `submit.sh`, `words.csv` | Mac | interim tool to run one batch from a CSV until the gateway exists |
| `DESIGN.md` | - | gateway design |

On the cluster, the service lives in `~/sd_service`: code, the SDXL model cache (`hf_cache/`, about 7 GB), job outputs and logs.
