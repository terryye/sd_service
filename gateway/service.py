"""The gateway's logic: validating tasks, the batcher, and the poller (DESIGN.md 3-6).

The batcher and poller are plain async "tick" methods; app.py calls them on a timer
and tests call them directly. One lock keeps the two from talking to the cluster at
the same time, so a batch is never submitted twice by them.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from .cluster import TERMINAL, ClusterBackend, ClusterError
from .config import Config
from .db import Batch, Database, Task

log = logging.getLogger("gateway")

NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
FIELDS = {"taskid", "prompt", "filename_prefix", "count", "on_duplicate", "postprocess", "seed", "style"}
IMAGE_SIZE = (1024, 1024)


def iso(t: float | None) -> str | None:
    if t is None:
        return None
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Service:
    def __init__(self, config: Config, backend: ClusterBackend, count_tokens: Callable[[str], int],
                 clock: Callable[[], float]):
        self.c = config
        self.backend = backend
        self.count_tokens = count_tokens
        self.clock = clock
        self.db = Database(config.db_path, clock)
        self.lock = asyncio.Lock()
        # Cluster reachability, for /api/health only; not worth persisting.
        self.last_ok: float | None = None
        self.failing_since: float | None = None

    # ---- submit -----------------------------------------------------------------

    def validate(self, raw) -> tuple[dict | None, str, str | None]:
        """Returns (task, on_duplicate, error) for one entry of a submit request."""
        if not isinstance(raw, dict):
            return None, "reject", "task must be an object"
        taskid = raw.get("taskid")
        if not isinstance(taskid, str) or not NAME.fullmatch(taskid):
            return None, "reject", "taskid must be 1-64 characters of A-Z a-z 0-9 _ -"
        unknown = sorted(set(raw) - FIELDS)
        if unknown:
            return None, "reject", f"unknown field: {', '.join(unknown)}"
        prompt = raw.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return None, "reject", "prompt must be a non-empty string"
        prefix = raw.get("filename_prefix", taskid)
        if not isinstance(prefix, str) or not NAME.fullmatch(prefix):
            return None, "reject", "filename_prefix must be 1-64 characters of A-Z a-z 0-9 _ -"
        count = raw.get("count", 1)
        if type(count) is not int or not 1 <= count <= 4:
            return None, "reject", "count must be an integer from 1 to 4"
        on_duplicate = raw.get("on_duplicate", "reject")
        if on_duplicate not in ("reject", "replace"):
            return None, "reject", "on_duplicate must be reject or replace"
        postprocess = raw.get("postprocess", "sticker")
        if postprocess not in ("sticker", "none"):
            return None, "reject", "postprocess must be sticker or none"
        seed = raw.get("seed")
        if seed is None:
            seed = secrets.randbelow(2**32)
        elif type(seed) is not int or not 0 <= seed < 2**32:
            return None, "reject", "seed must be an integer from 0 to 4294967295"
        style = raw.get("style")
        if style is not None and style not in self.c.styles:
            names = ", ".join(sorted(self.c.styles)) or "none configured"
            return None, "reject", f"unknown style {style!r} (available: {names})"
        # The style's trigger words are part of what SDXL reads, so they count too.
        full = self.c.styles[style].full_prompt(prompt.strip()) if style else prompt.strip()
        tokens = self.count_tokens(full)
        if tokens > self.c.max_prompt_tokens:
            return None, "reject", f"prompt too long: {tokens} tokens, at most {self.c.max_prompt_tokens}"
        task = {"taskid": taskid, "prompt": prompt.strip(), "filename_prefix": prefix,
                "count": count, "postprocess": postprocess, "seed": seed, "style": style}
        return task, on_duplicate, None

    def submit(self, entries: list) -> list[dict]:
        answers = []
        for raw in entries:
            task, on_duplicate, error = self.validate(raw)
            if error:
                taskid = raw.get("taskid") if isinstance(raw, dict) else None
                answers.append({"taskid": taskid, "result": "rejected", "error": error})
                continue
            answer, replaced = self.db.submit(task, on_duplicate)
            if replaced is not None:
                shutil.rmtree(self.c.results_dir / replaced.taskid, ignore_errors=True)
            answers.append(answer)
        return answers

    # ---- status -----------------------------------------------------------------

    def task_view(self, task: Task) -> dict:
        view = {"taskid": task.taskid, "result": task.result, "version": task.version,
                "created": iso(task.created), "updated": iso(task.updated),
                "images": [{"url": f"/results/{task.taskid}/{i['file']}?v={task.version}",
                            "seed": i["seed"], "result": i["result"]} for i in task.images]}
        if task.style:
            view["style"] = task.style
        if task.error:
            view["error"] = task.error
        return view

    def health(self) -> dict:
        now = self.clock()
        if self.failing_since is None:
            cluster = "reachable" if self.last_ok is not None else "unknown"
        elif now - self.failing_since > self.c.unreachable_after_s:
            cluster = "unreachable"
        else:
            cluster = "reachable" if self.last_ok is not None else "unknown"
        disk_free = shutil.disk_usage(self.c.data_dir).free / 1e9
        degraded = cluster == "unreachable" or disk_free < self.c.min_disk_free_gb
        return {"status": "degraded" if degraded else "ok", "cluster": cluster,
                "last_poll": iso(self.last_ok), **self.db.queue_stats(),
                "batches_in_flight": len(self.db.in_flight()),
                "disk_free_gb": round(disk_free, 1), "styles": sorted(self.c.styles)}

    # ---- batcher ------------------------------------------------------------------

    async def batcher_tick(self) -> Batch | None:
        """Form and submit a batch when none is in flight and a limit is hit (DESIGN.md 5)."""
        async with self.lock:
            if self.db.in_flight():
                return None
            queued = self.db.queued()
            if not queued:
                return None
            total = sum(t.count for t in queued)
            waited = self.clock() - min(t.queued_at for t in queued)
            if total < self.c.max_batch_images and waited < self.c.max_wait_s:
                return None
            chosen, images = [], 0
            for t in queued:  # first in, first out; a task is never split
                if images + t.count > self.c.max_batch_images:
                    break
                chosen.append(t.taskid)
                images += t.count
            for t in queued:
                if t.taskid in chosen and t.style and t.style not in self.c.styles:
                    # The style was removed from config.toml after the task was queued.
                    self.db.fail_task(t.taskid, f"style {t.style!r} is no longer configured")
                    chosen.remove(t.taskid)
            if not chosen:
                return None
            batch = self.db.claim_batch(self.c.batch_prefix, chosen)
            log.info("batch %s: %d tasks, %d images", batch.batch_id, len(chosen), images)
            await self._try_submit(batch)
            return self.db.get_batch(batch.batch_id)

    def batch_payload(self, batch: Batch) -> dict:
        tasks = self.db.batch_tasks(batch.batch_id)
        styles = {}
        for name in sorted({t.style for t in tasks if t.style}):
            s = self.c.styles[name]
            styles[name] = {"lora": s.lora_file, "scale": s.scale,
                            "embeddings": s.embeddings_file or None, "tokens": list(s.embedding_tokens)}
        return {"batch_id": batch.batch_id, "styles": styles, "tasks": [
            {"taskid": t.taskid, "version": t.version, "count": t.count, "seeds": t.seeds(),
             "prompt": self.c.styles[t.style].full_prompt(t.prompt) if t.style else t.prompt,
             "style": t.style, "postprocess": t.postprocess, "filename_prefix": t.filename_prefix}
            for t in tasks]}

    async def _try_submit(self, batch: Batch) -> None:
        """Upload batch.json and sbatch. A failure here never counts as an attempt."""
        self.db.mark_submit_tried(batch.batch_id)
        payload = self.batch_payload(batch)
        try:
            await self.backend.upload_batch(batch.batch_id, payload)
            job_id = await self.backend.submit(batch.batch_id, sum(t["count"] for t in payload["tasks"]))
        except ClusterError as exc:
            self._cluster_failed()
            log.warning("batch %s: submit failed, will look it up before resubmitting: %s",
                        batch.batch_id, exc)
            return
        self._cluster_ok()
        self.db.mark_submitted(batch.batch_id, job_id)
        log.info("batch %s: submitted as job %s", batch.batch_id, job_id)

    # ---- poller ---------------------------------------------------------------------

    async def poller_tick(self) -> None:
        """One heartbeat: check every in-flight batch, or just the cluster when idle."""
        async with self.lock:
            batches = self.db.in_flight()
            if not batches:
                try:
                    await self.backend.ping()
                    self._cluster_ok()
                except ClusterError:
                    self._cluster_failed()
                return
            for batch in batches:
                try:
                    await self._poll_batch(batch)
                    self._cluster_ok()
                except ClusterError as exc:
                    self._cluster_failed()
                    log.warning("batch %s: poll failed: %s", batch.batch_id, exc)

    async def _poll_batch(self, batch: Batch) -> None:
        result = await self.backend.poll(batch.batch_id)
        worker = result.worker
        if batch.state == "submitting":
            if worker is not None:
                # An sbatch that reported failure went through after all.
                self.db.mark_submitted(batch.batch_id, worker[0])
            elif batch.submit_tried is None or self.clock() - batch.submit_tried >= self.c.resubmit_after_s:
                await self._try_submit(batch)
                return
            else:
                return
        if worker is None:
            return  # sacct has not listed the job yet
        job_id, state = worker
        self.db.set_job_state(batch.batch_id, job_id, state)

        running = {t.taskid: t for t in self.db.batch_tasks(batch.batch_id) if t.result == "running"}
        new = [r for r in result.progress
               if r["taskid"] in running and r["version"] == running[r["taskid"]].version]
        if new:
            await self._collect(batch, new, running)

        if state in TERMINAL:
            for task in self.db.batch_tasks(batch.batch_id):
                if task.result == "running":
                    self.db.charge_attempt(task.taskid, batch.batch_id,
                                           f"job {job_id} ended ({state}) before the task finished",
                                           self.c.max_attempts)
            await self.backend.cleanup(batch.batch_id)
            shutil.rmtree(self.c.staging_dir / batch.batch_id, ignore_errors=True)
            self.db.finish_batch(batch.batch_id)
            log.info("batch %s: done, job %s %s", batch.batch_id, job_id, state)

    async def _collect(self, batch: Batch, records: list[dict], running: dict[str, Task]) -> None:
        """Pull finished tasks' images, check them, and mark the tasks done."""
        staging = self.c.staging_dir / batch.batch_id
        if any("images" in r for r in records):
            await self.backend.fetch(batch.batch_id, staging)
        for rec in records:
            task = running[rec["taskid"]]
            if "error" in rec:
                self.db.charge_attempt(task.taskid, batch.batch_id, f"generation failed: {rec['error']}",
                                       self.c.max_attempts)
                continue
            src = staging / task.taskid / f"v{task.version}"
            try:
                images = self._place_images(task, src, rec["images"])
            except (OSError, ValueError) as exc:
                # Not fully synced yet, or broken: try again on the next poll.
                log.warning("batch %s: task %s not collected yet: %s", batch.batch_id, task.taskid, exc)
                continue
            self.db.complete_task(task.taskid, task.version, batch.batch_id, images)

    def _place_images(self, task: Task, src: Path, listed: list[dict]) -> list[dict]:
        expected = [f"{task.filename_prefix}_{n}.webp" for n in range(1, task.count + 1)]
        if sorted(i["file"] for i in listed) != sorted(expected):
            raise ValueError(f"expected {expected}, progress lists {[i['file'] for i in listed]}")
        for name in expected:
            with Image.open(src / name) as im:
                if im.format != "WEBP" or im.size != IMAGE_SIZE:
                    raise ValueError(f"{name}: {im.format} {im.size}")
                im.verify()
        dest = self.c.results_dir / task.taskid
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        for name in expected:
            shutil.copyfile(src / name, dest / name)
        by_file = {i["file"]: i for i in listed}
        return [{"file": n, "seed": by_file[n]["seed"], "result": by_file[n]["result"]} for n in expected]

    # ---- reachability ---------------------------------------------------------------

    def _cluster_ok(self) -> None:
        self.last_ok = self.clock()
        self.failing_since = None

    def _cluster_failed(self) -> None:
        if self.failing_since is None:
            self.failing_since = self.clock()
