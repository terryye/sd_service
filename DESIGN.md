# Image Generation Gateway: Design

Status: implemented (gateway/, cluster/). Last updated 2026-11-05.

## 1. Purpose

A small web service that lets the dictionary app's backend request kids' illustrations, generated with SDXL on Northeastern's Explorer HPC cluster. Clients submit tasks over HTTP and poll for results. The gateway queues the tasks, batches them into Slurm GPU jobs, tracks the jobs, and serves the finished images.

**Goals:** simple API, durable queue, no lost tasks across restarts, efficient GPU use, safe for children's content.

**Non-goals:**
- **Real-time latency.** A task takes minutes: Slurm queue wait plus model load plus generation.
- **More than one gateway process.**
- **Image editing, or choosing between candidates.** The client picks; the gateway only generates and serves.
- **Deleting images automatically.**

## 2. Architecture

```
dictionary app backend ──HTTP──> Gateway (one process, always-on server, private address)
                                              ├─ API         submit, status, bulk status, health, result files
                                              ├─ SQLite      tasks (also the queue), batches
                                              ├─ Batcher     queued tasks -> batch -> sbatch
                                              └─ Poller      every 30 s: job state + progress (heartbeat)
                                                        │  outbound SSH / rsync only
                                                        ▼
                                              Explorer: sd_job.sh -> generate.py
                                              (batch.json in; images + progress.jsonl out)
```

- **All connections start at the gateway.** The cluster never contacts the gateway or a client.
- **The cluster doesn't need a public address, and neither do clients.** Only the gateway's API must be reachable by its clients.
- **One Slurm job serves many tasks,** because loading the model costs 1-2 minutes per job.

## 3. API

This is an internal service. Only the dictionary app's backend calls it, and app users never reach it directly. It listens on a private address and has no authentication.

### 3.1 Submit tasks

`POST /api/images/gen`

```json
{"tasks": [
  {"taskid": "apple-01", "prompt": "a shiny red apple, cartoon sticker",
   "filename_prefix": "apple", "count": 4, "on_duplicate": "reject"},
  {"taskid": "cat-01", "prompt": "a small orange cat, cartoon sticker", "on_duplicate": "replace"}
]}
```

| Field | Required | Default | Rules |
|---|---|---|---|
| `taskid` | yes | - | globally unique; `[A-Za-z0-9_-]`, 1-64 characters |
| `prompt` | yes | - | at most 77 tokens, SDXL's limit (about 50-60 words; see 7); the kids' safety negative prompt is always added |
| `filename_prefix` | no | the taskid | `[A-Za-z0-9_-]`, 1-64 characters; files are `<prefix>_<n>.webp` |
| `count` | no | 1 | 1-4 candidate images |
| `on_duplicate` | no | `reject` | `reject` or `replace` (section 4.2) |
| `postprocess` | no | `sticker` | `sticker` (white background + pattern retry) or `none` |
| `seed` | no | random | seed for image 1; images 2..N derive their seeds from it. Similar, not identical, results across GPU types (see 12) |
| `style` | no | none | a LoRA style configured in `config.toml` (7.1); its trigger words are added in front of the prompt |

**Response:** `202`, with one entry per task, in request order. Each task is accepted or rejected on its own:

```json
{"tasks": [
  {"taskid": "apple-01", "result": "queued", "version": 1},
  {"taskid": "cat-01", "result": "queued", "version": 3, "replaced": true},
  {"taskid": "dog-01", "result": "rejected", "error": "taskid exists"}
]}
```

- **Whole-request errors:**
  - `400`: malformed JSON, or more than 100 tasks.
- **Per-task rejections** (`result: "rejected"`): an invalid field, an unknown `style`, `prompt too long` (over 77 tokens, counting the style's trigger words), `taskid exists` (with `on_duplicate: reject`), or `task is running` (a `replace` of a running task).

### 3.2 Status of one task

`GET /api/images/tasks/{taskid}`

```json
{"taskid": "apple-01", "result": "success", "version": 1,
 "created": "2026-11-05T14:00:02Z", "updated": "2026-11-05T14:06:40Z",
 "images": [
   {"url": "/results/apple-01/apple_1.webp?v=1", "seed": 99812, "result": "success"},
   {"url": "/results/apple-01/apple_2.webp?v=1", "seed": 31007, "result": "success"},
   {"url": "/results/apple-01/apple_3.webp?v=1", "seed": 55120, "result": "needs_review"},
   {"url": "/results/apple-01/apple_4.webp?v=1", "seed": 70411, "result": "success"}]}
```

This returns `404` if the taskid doesn't exist. While a task is `queued` or `running`, `images` is empty. A `failed` task has an `error` field.

### 3.3 Bulk status

`GET /api/images/tasks?ids=apple-01,cat-01,dog-99`

```json
{"tasks": [
  {"taskid": "apple-01", "result": "success", "version": 1, "images": [...]},
  {"taskid": "cat-01", "result": "running", "version": 3, "images": []},
  {"taskid": "dog-99", "result": "not_found"}
]}
```

At most 100 ids per call; more gives a `400`. Entries come back in the order requested, with the same objects as 3.2. Unknown ids come back as `not_found` rather than failing the whole call.

### 3.4 Health

`GET /api/health`

```json
{"status": "ok", "cluster": "reachable", "last_poll": "2026-11-05T14:02:30Z",
 "queued_tasks": 12, "queued_images": 40, "oldest_queued_age_s": 45,
 "batches_in_flight": 1, "disk_free_gb": 182.4, "styles": ["icon3d"]}
```

`styles` lists the configured style names. `status` is `degraded` when the cluster has been unreachable for more than 5 minutes, or when free disk falls below a configured threshold.

### 3.5 Result files

`GET /results/{taskid}/{prefix}_{n}.webp?v={version}`

- **Format:** 1024×1024 WebP, quality 90, about 60 KB each.
- **Access:** open, like the rest of the API. The app backend downloads the images and serves them to its own users.
- **The `v` parameter** only defeats caches after a replace. The server always returns the current version.

### 3.6 Test page

`GET /` serves a single static HTML page for trying the service by hand. It's plain HTML and JavaScript with no build step, served by the gateway itself, so it calls the API on the same origin.

- **Form:** a prompt box; `count` (1-4); an optional `filename_prefix`; a style (from `GET /api/health`); `postprocess` (sticker or none); and `on_duplicate`. The taskid is filled in as `test-<timestamp>` and can be edited. Submit sends `POST /api/images/gen` and shows the per-task answer, queued or the rejection reason.
- **Task list:** every task submitted from this browser, newest first. The list of taskids is kept in the browser's `localStorage`, so it survives a page reload. It refreshes every 5 seconds with one bulk status call, and stops refreshing once every task has finished.
- **Each row:** taskid, prompt, a status badge (`queued`, `running`, `success`, `needs_review`, `failed`), and the time since submission. A failed task shows its `error`.
- **Preview:** once a task is done, its candidates show as thumbnails, each marked `success` or `needs_review`. Clicking one opens the full 1024×1024 image.
- **Health strip** at the top, from `GET /api/health`: cluster reachable or not, queue length, batches in flight, and last poll time.

It's a testing aid with no state of its own; everything it shows comes from the API.

## 4. Task model

### 4.1 States

```
queued ──> running ──> success        (at least one image "success")
   ^          │   ├──> needs_review   (images exist, but all failed the pattern check)
   │          │   └──> failed         (generation failed 3 times; error says why)
   └──────────┘  batch ended before this task finished: back to queued (see 6.3)
```

Each image has its own `result`: `success` or `needs_review`. `needs_review` means the image still looked like a repeating pattern, or had an object cut off by the edge, after 4 tries. The image exists, but a person should look at it before using it.

### 4.2 Duplicate taskids (`on_duplicate`)

- **`reject`** (default): if the taskid exists, the task is rejected. The existing task is unchanged.
- **`replace`:** the existing task is overwritten with the new fields, and its `version` goes up by 1. What happens depends on the old task's state:

| Old state | Replace does |
|---|---|
| `queued` | updates the fields in place; the task stays in its queue position |
| `running` | rejected with `task is running`; the client resubmits after the task finishes |
| `success` / `needs_review` / `failed` | deletes the old images, resets `attempts`, and sets the task to `queued` |

The `version` only changes the result URLs (`?v=`), so caches fetch the new images.

## 5. Queue and batching

**The queue is the `tasks` table in SQLite.** A task's state is its place in the queue, so the queue and the status store are always the same data.

```sql
tasks(taskid TEXT PRIMARY KEY, prompt TEXT, filename_prefix TEXT,
      count INT, postprocess TEXT, seed INT, version INT, result TEXT,
      images JSON, error TEXT, attempts INT, batch_id TEXT,
      created TEXT, updated TEXT)
batches(batch_id TEXT PRIMARY KEY, slurm_job_id TEXT, job_name TEXT, state TEXT,
        tasks JSON,  -- [taskid, ...]
        submitted TEXT, last_progress TEXT, polled TEXT)
```

**The batcher** checks every 5 seconds. It forms a batch when:
- **No batch is in flight.** Only one Slurm job runs at a time. A full batch of 200 images takes about 30 minutes of generation, which is plenty for a dictionary.
- **And either limit is hit:** the queued tasks add up to 200 images, or the oldest queued task has waited 60 seconds.

How a batch is filled:
- **Order:** first in, first out.
- **Size:** at most 200 images, counted as the sum of `count`. A task is never split across batches.
- **Claiming:** in one transaction, the batcher sets the chosen tasks to `running`, records their `batch_id`, and inserts the batch row. Then it uploads and submits (section 6).

**Durability:** after a restart, the gateway reloads queued tasks and resumes polling in-flight batches from the `batches` table. Nothing is held only in memory.

**Scaling limit:** this design assumes exactly one gateway process. Several processes would need PostgreSQL (`FOR UPDATE SKIP LOCKED`) or a message broker.

## 6. Cluster interaction

### 6.1 The contract with the job

Paths below are relative to the environment's directory on the cluster, `~/sd_service/<env>/` (see 6.4). The gateway uploads `batches/<batch_id>/batch.json`:

```json
{"batch_id": "sdg-prod-000123",
 "styles": {"icon3d": {"lora": "icon3d-4623a4b46c63.safetensors", "scale": 1.0,
                       "embeddings": "icon3d-8707994599a5.emb.safetensors", "tokens": ["<s0>", "<s1>"]}},
 "tasks": [
  {"taskid": "apple-01", "version": 1, "prompt": "3d icon in the style of <s0><s1>, ...", "style": "icon3d",
   "count": 4, "seeds": [99812, 31007, 55120, 70411], "postprocess": "sticker", "filename_prefix": "apple"}]}
```

The job writes `out/<batch_id>/<taskid>/v<version>/<prefix>_<n>.webp`. After each task it appends one line to `out/<batch_id>/progress.jsonl`:

```json
{"taskid": "apple-01", "version": 1, "ts": "...",
 "images": [{"file": "apple_1.webp", "seed": 99812, "result": "success", "attempts": 1}, ...]}
```

`prompt` is the full prompt the model reads, trigger words included. `styles` holds only the styles this batch's tasks use. Generation is resumable: a re-run skips tasks that already appear in `progress.jsonl`.

### 6.2 Submitting

- **Slurm settings:** `sbatch --job-name=<batch_id> --dependency=singleton`, on the `gpu` partition with 1 GPU. The time limit is 30 minutes plus 15 seconds per image, capped at 8 hours. Start-up alone has taken from under 2 to over 11 minutes depending on the node, and one job with a 10-minute base timed out before generating anything.
- **Failed submissions:** `sbatch` can report a failure ("Unexpected message received") and still create the job, sometimes minutes later. So the gateway never resubmits blindly:
  - every submission of a batch reuses the same job name;
  - `--dependency=singleton` stops two copies from running at once;
  - `sd_job.sh` exits immediately unless it is the lowest job ID with that name.

  After a failed `sbatch`, the gateway looks up the batch's job name with `sacct` for up to 5 minutes, and only then submits again.

### 6.3 Poller (heartbeat)

The poller runs every 30 seconds while a batch is in flight, as one short SSH command:

```
sacct -n -X -P -u <user> -S now-7days --name=<batch_id> -o JobID,State ; tail progress.jsonl
```

| What the poll sees | What the gateway does |
|---|---|
| SSH fails | retries; `cluster: unreachable` after 5 minutes of failures |
| `PENDING` | nothing; tasks stay `running` (handed to the cluster) |
| `RUNNING`, new progress lines | pulls the finished tasks' images, marks those tasks done |
| `COMPLETED` | pulls the remaining images; any task missing from progress is requeued |
| `FAILED` / `TIMEOUT` / `NODE_FAIL` / `CANCELLED` | unfinished tasks are requeued |

**Requeueing and attempts.** Attempts count failures of *generation*, not of waiting:
- **Queue wait (`PENDING`):** never counts, and has no time limit.
- **Cluster unreachable, or `sbatch` errors:** never count.
- **A batch that ends while unfinished tasks remain:** each unfinished task goes back to `queued` with `attempts + 1`, or becomes `failed` when `attempts` reaches 3.
- **A hung job** is ended by its Slurm time limit, and is then handled the same way.
- **Pulling:** rsync from the transfer node, then check each file (it decodes as WebP, 1024×1024), then move it into `data/results/<taskid>/`. Only then is the task marked done.
- **Cleanup:** once a whole batch has been collected, the gateway deletes `batches/<batch_id>` and `out/<batch_id>` on the cluster. The gateway's disk is the permanent store, and the cluster home quota is 75 GB.

### 6.4 Cluster-side environment

```
~/sd_service/
  pytorch.sif        pinned copy of RC's PyTorch container (shared by all environments)
  hf_cache/          SDXL model weights, about 7 GB (shared)
  loras/             LoRA style files, named <style>-<first 12 hex of sha256> (shared)
  dev/  prod/        one directory per environment: cluster code, batches/, out/, logs/
```

- **One directory per environment.** The dev gateway (on the Mac) and the prod gateway (on the server) must not see each other's batches: they would overwrite each other's files, and the "lowest job ID with this name wins" rule would mix their jobs. So each environment has its own directory and its own batch and job-name prefix (`sdg-dev-…`, `sdg-prod-…`), set by `env` in the config. Each also has its own copy of the cluster code, so testing new code in dev never touches prod.
- **A pinned container.** RC can replace `/shared/container_repository/explorer/pytorch/pytorch_training.sif` at any time (it last changed on Oct 20, 2026), and a new torch or transformers inside it could break the pinned `diffusers`. The service runs from its own copy, `~/sd_service/pytorch.sif` (4.3 GB, one file, so shared storage handles it well). It is refreshed only on purpose, after testing in dev.
- **Python packages:** `diffusers==0.42.1`, `peft==0.21.2` (for LoRAs) and their few missing dependencies are installed into node-local `/tmp` in each job.
- **Disk:** the shared files take about 11 GB of the 75 GB home quota. Batch output is temporary and is removed after the gateway pulls it.

## 7. Image pipeline

- **Model:** SDXL base 1.0 with the fp16-fix VAE, 30 steps, guidance 6.0, always 1024×1024.
- **Prompt length:** SDXL's text encoders read at most 77 tokens and silently ignore the rest. The gateway counts tokens at submit time with SDXL's CLIP tokenizer and rejects longer prompts with `prompt too long`, so a prompt is never cut off without the client knowing.
- **Negative prompt:** the kids' safety negative prompt is always added to the client's prompt.
- **`postprocess: sticker`:**
  - whiten the background: the edge-connected flat background, holes that match it, stray specks, and the light grey haze of a soft shadow around the sticker;
  - check for a repeating pattern or an object cut off by the image edge, and if found, retry with the next seed, up to 4 tries;
  - if all tries fail, keep the image and mark it `needs_review`.
- **`postprocess: none`:** the raw SDXL output.
- **Saving:** all checks run on the full-quality image; the file is saved last, as WebP at quality 90.

### 7.1 Styles (LoRAs)

A style is a LoRA, configured once in `config.toml` and picked per task with `"style": "<name>"`. Adding a LoRA is a config change and a `deploy`, not a code change.

```toml
[[styles]]
name = "icon3d"                                       # what tasks ask for
source = "hf:linoyts/3d-icon-sdxl-lora/pytorch_lora_weights.safetensors"   # or "civitai:<version id>"
sha256 = "4623a4b4..."                                # required; deploy refuses a file that doesn't match
trigger = "3d icon in the style of <s0><s1>"          # added in front of the prompt
scale = 1.0                                           # LoRA strength, 0-2
embeddings = "hf:linoyts/3d-icon-sdxl-lora/embeddings.safetensors"         # optional, pivotal tuning
embeddings_sha256 = "87079945..."
embedding_tokens = ["<s0>", "<s1>"]
```

- **Download:** `deploy` fetches each file on the transfer node into `~/sd_service/loras/`, checks its SHA-256, and skips files already there. Civitai downloads use `[downloads] civitai_token`, sent over ssh's stdin so it never appears on a command line on the cluster.
- **Trigger words** are part of the prompt the model reads, so the 77-token limit counts them. The gateway's tokenizer knows each style's embedding tokens, so `<s0>` counts as one token, as it does in the job.
- **In the job:** only the styles the batch uses are loaded, once, after the model; each task then switches the active LoRA (or turns LoRAs off for a plain task). Embeddings must be in the diffusers SDXL format: one safetensors file with `clip_l` and `clip_g` tensors.
- **Removing a style** from the config fails its still-queued tasks with `style '<name>' is no longer configured`.
- **Licenses** differ per LoRA (some require crediting the creator); check each one before using it in the app.

## 8. Security

**Input validation:**
- Taskid and prefix are limited to `[A-Za-z0-9_-]`, so no request can build a path like `../`.
- The server builds every file path itself.

**SSH from the gateway to Explorer:** the deploy server is treated as a trusted environment. The gateway uses a dedicated SSH key with no extra restrictions for now. Deleting that key's line in `~/.ssh/authorized_keys` on Explorer revokes its access.

**Network:** the service listens on a private address that only the app backend can reach. It is never exposed to app users or the internet.

## 9. Configuration

One file, `config.toml`, so the same code runs on a laptop and on the server:

```toml
[server]    listen = "127.0.0.1:8080"   data_dir = "./data"
[cluster]   user = "ye.te"   login_host = "login.explorer.northeastern.edu"
            xfer_host = "xfer.discovery.neu.edu"   ssh_key = "~/.ssh/sd_gateway"   remote_dir = "~/sd_service"
            env = "dev"   # "prod" on the server; picks ~/sd_service/<env>/ and the sdg-<env>- prefix
[queue]     max_batch_images = 200   max_wait_s = 60   max_attempts = 3
[poll]      interval_s = 30   unreachable_after_s = 300
[limits]    max_tasks_per_request = 100   max_ids_per_status = 100   min_disk_free_gb = 10
[downloads] civitai_token = ""          # deploy only
[[styles]]  ...                         # one table per LoRA style, see 7.1
```

## 10. Implementation layout and testing

```
gateway/
  app.py        FastAPI routes (submit, status, bulk status, health, results, test page)
  static/index.html   the test page (3.6)
  service.py    task validation, the batcher and the poller (heartbeat, collecting, retries)
  db.py         SQLite schema, queue operations, state transitions
  cluster.py    ClusterBackend interface and SlurmBackend (SSH, rsync)
  config.py     config.toml
  __main__.py   `serve` and `deploy` commands
cluster/        sd_job.sh, generate.py, postprocess.py (deployed to ~/sd_service/<env>/)
tests/          FakeBackend (fake_cluster.py) and the unit tests
```

- **Stack:** Python 3.12, FastAPI with uvicorn, and the `sqlite3` module from the standard library. The batcher and poller are background tasks in the same process.
- **Unit tests** use `FakeBackend`, with no GPU or SSH. They cover:
  - validation and per-task rejection, including prompts over 77 tokens
  - dev and prod settings never touch each other's cluster directories or job names
  - duplicate `reject` and `replace` in every state, including rejecting a replace while running
  - batch formation (size and time limits, one batch at a time, first-in-first-out order)
  - collecting results while a batch is running
  - attempts: failed batches charge their unfinished tasks; queue wait never counts
  - failed and timed-out jobs
  - recovery after a restart
  - bulk status order and `not_found`
- **End-to-end test** against Explorer: several overlapping requests, from `curl` and the test page, including a replace of a finished task. Check that the images are correct and that each batch ran as exactly one Slurm job.

## 11. Deployment

1. Develop and test on the Mac with `listen = 127.0.0.1`.
2. Move to the always-on server:
   - copy the code, `config.toml` and the dedicated SSH key;
   - run as a systemd service, listening on a private address;
   - back up `data/` (the SQLite file and `results/`). It is the only copy of the images.
3. Watch `GET /api/health`: cluster reachability, queue age and free disk.

## 12. Notes

- **Seeds are not exactly reproducible across GPU types.** The same prompt and seed on a V100 and on an H200 can give slightly different images. A `seed` gives similar results, not identical files.
- **Usage counts against the account's fair share.** Every batch lowers the Slurm priority of the account it runs under, including the owner's own coursework jobs. At dictionary scale this is minor, but heavy use makes all of that account's jobs wait longer.
- **Model license.** SDXL base 1.0 is released under the CreativeML Open RAIL++-M license. It allows using the generated images in an app, but forbids certain uses; the course report, and the app if it is ever published, should mention it.

## 13. Open items

- **RC check:** before going live, confirm with the course instructor or RC (`rchelp@northeastern.edu`) that an automated service using a personal Explorer account for coursework is acceptable.
