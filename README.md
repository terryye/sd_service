# sd_service

An internal image generation service for a kids' dictionary app. The app's backend submits prompts over HTTP; the service generates sticker-style illustrations (one subject, pure white background, 1024×1024 WebP) with SDXL on GPUs of Northeastern's Explorer HPC cluster, and serves the results.

```
app backend ──HTTP──> gateway ──SSH, rsync──> Explorer: Slurm GPU job (SDXL)
```

The gateway runs outside the cluster: it queues tasks in SQLite, batches them into Slurm GPU jobs (up to 200 images, or after 60 s), polls the job every 30 s, and serves the finished images. All connections start at the gateway; the cluster never calls back. [DESIGN.md](DESIGN.md) has the full design.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/images/gen` | submit up to 100 tasks; each is queued or rejected |
| `GET /api/images/tasks/{taskid}` | status and image URLs of one task |
| `GET /api/images/tasks?ids=a,b,c` | bulk status |
| `GET /api/health` | cluster reachability, queue length, free disk |
| `GET /results/{taskid}/{prefix}_{n}.webp` | an image |
| `GET /` | test page: submit a prompt, watch the status, preview the images |

A task is one prompt (at most 77 tokens) with 1 to 4 candidate images:

```bash
curl -X POST http://127.0.0.1:8080/api/images/gen -H 'Content-Type: application/json' -d '{"tasks": [
  {"taskid": "apple-01", "prompt": "a cute cartoon sticker of a red apple, single object, white background",
   "filename_prefix": "apple", "count": 2}]}'
curl http://127.0.0.1:8080/api/images/tasks/apple-01
```

Optional fields: `on_duplicate` (`reject` or `replace`), `postprocess` (`sticker` or `none`), `seed`, and `style`: a LoRA configured as a `[[styles]]` entry in `config.toml` (see `config.example.toml` and DESIGN.md 7.1). `deploy` downloads and checks each style's files.

## Running it

Requires Python 3.11+, and SSH access to Explorer with a key that works without a password.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -e '.[dev]'
cp config.example.toml config.toml      # set user, env ("dev" or "prod"), listen address
.venv/bin/python -m gateway deploy      # copy cluster/ to ~/sd_service/<env>/, pin the container
.venv/bin/python -m gateway serve       # then open http://127.0.0.1:8080/
.venv/bin/python -m pytest              # unit tests, no cluster needed
```

Run `deploy` again whenever `cluster/` changes. Each environment (`dev`, `prod`) has its own folder and job names on the cluster, so they never touch each other. `data/` holds the SQLite file and every image; it is the only copy, so back it up.

## Files

| Path | What it is |
|---|---|
| `gateway/` | the HTTP service: API, SQLite queue, batcher, poller, test page |
| `cluster/` | runs on Explorer: `sd_job.sh` (Slurm job), `generate.py` (SDXL), `postprocess.py` (white background, pattern and cut-off checks) |
| `tests/` | unit tests with a fake cluster |
| `DESIGN.md` | the design |

On the cluster, `~/sd_service/` holds the pinned PyTorch container, the SDXL model cache (about 7 GB), and one folder per environment.
