"""An in-memory stand-in for Explorer, used by the tests (DESIGN.md 10)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from gateway.cluster import ClusterError, PollResult, WarmWorker


class FakeBackend:
    def __init__(self, remote: Path):
        self.remote = remote  # plays the cluster's out/ directory
        self.uploads: dict[str, dict] = {}
        self.submits: list[str] = []
        self.jobs: dict[str, list[tuple[str, str]]] = {}
        self.progress: dict[str, list[dict]] = {}
        self.cleaned: list[str] = []
        self.next_job = 1000
        self.down = False  # every call fails, like an SSH outage
        self.submit_fails: list[bool] = []  # per submit: True = sbatch errors; queued up front
        self.ghost = False  # a failing sbatch still creates the job
        self.nameless: dict[str, list[tuple[str, str]]] = {}  # jobs Slurm lost the name of, by job ID
        self.polled_with: list[str | None] = []
        self.cancelled: list[str] = []
        self.warm: list[WarmWorker] = []  # workers/<job>.json files
        self.inbox: dict[str, str] = {}  # inbox/<job> -> batch id
        self.claimed: set[str] = set()  # out/<batch>/claimed
        self.finished: set[str] = set()  # out/<batch>/finished
        self.hand_off_fails = False

    def _check(self) -> None:
        if self.down:
            raise ClusterError("ssh: connect timed out")

    async def ping(self) -> None:
        self._check()

    async def upload_batch(self, batch_id: str, payload: dict) -> None:
        self._check()
        self.uploads[batch_id] = json.loads(json.dumps(payload))

    async def submit(self, batch_id: str, n_images: int) -> str:
        self._check()
        self.submits.append(batch_id)
        fail = self.submit_fails.pop(0) if self.submit_fails else False
        if fail and not self.ghost:
            raise ClusterError("sbatch: error: Batch job submission failed: Unexpected message received")
        self.next_job += 1
        job_id = str(self.next_job)
        self.jobs.setdefault(batch_id, []).append((job_id, "PENDING"))
        if fail:
            raise ClusterError("sbatch: error: Batch job submission failed: Unexpected message received")
        return job_id

    def _job(self, job_id: str | None) -> list[tuple[str, str]]:
        """sacct -j: a job by ID, whatever batch name it was submitted under."""
        found = [j for jobs in self.jobs.values() for j in jobs if j[0] == job_id]
        return found or self.nameless.get(job_id, [])

    async def poll(self, batch_id: str, job_id: str | None = None) -> PollResult:
        self._check()
        self.polled_with.append(job_id)
        jobs = dict(self.jobs.get(batch_id, []) + self._job(job_id))
        return PollResult(sorted(jobs.items(), key=lambda j: int(j[0])),
                          [dict(r) for r in self.progress.get(batch_id, [])],
                          claimed=batch_id in self.claimed, finished=batch_id in self.finished)

    async def fetch(self, batch_id: str, dest: Path) -> None:
        self._check()
        src = self.remote / batch_id
        if src.exists():
            shutil.copytree(src, dest, dirs_exist_ok=True)

    async def cleanup(self, batch_id: str, cancel: list[str]) -> None:
        self._check()
        self.cleaned.append(batch_id)
        self.cancelled += cancel
        shutil.rmtree(self.remote / batch_id, ignore_errors=True)

    async def warm_workers(self) -> list[WarmWorker]:
        self._check()
        return list(self.warm)

    async def hand_off(self, job_id: str, batch_id: str) -> None:
        self._check()
        if self.hand_off_fails:
            raise ClusterError("ssh: connection reset")
        self.inbox[job_id] = batch_id

    async def take_back(self, job_id: str) -> bool:
        self._check()
        return self.inbox.pop(job_id, None) is not None

    # ---- what the warm worker would do ----------------------------------------------

    def go_warm(self, job_id: str, idle_until: float, job_end: float) -> None:
        self.warm = [w for w in self.warm if w.job_id != job_id] + [WarmWorker(job_id, idle_until, job_end, "RUNNING")]

    def claim(self, job_id: str) -> str:
        batch_id = self.inbox.pop(job_id)
        self.claimed.add(batch_id)
        self.warm = [w for w in self.warm if w.job_id != job_id]
        return batch_id

    def finish(self, batch_id: str) -> None:
        self.finished.add(batch_id)

    # ---- what the Slurm job would do ------------------------------------------------

    def set_state(self, batch_id: str, state: str) -> None:
        self.jobs[batch_id][0] = (self.jobs[batch_id][0][0], state)

    def finish_task(self, batch_id: str, taskid: str, results: list[str] | None = None,
                    size=(1024, 1024)) -> None:
        """Write a task's images and its progress line, as generate.py does."""
        task = next(t for t in self.uploads[batch_id]["tasks"] if t["taskid"] == taskid)
        results = results or ["success"] * task["count"]
        out = self.remote / batch_id / taskid / f"v{task['version']}"
        out.mkdir(parents=True, exist_ok=True)
        images = []
        for n, (seed, result) in enumerate(zip(task["seeds"], results, strict=True), start=1):
            name = f"{task['filename_prefix']}_{n}.webp"
            Image.new("RGB", size, (255, 255, 255)).save(out / name, "WEBP", quality=90)
            images.append({"file": name, "seed": seed, "result": result, "attempts": 1})
        self.progress.setdefault(batch_id, []).append(
            {"taskid": taskid, "version": task["version"], "images": images, "ts": "2026-11-05T00:00:00Z"})

    def fail_task(self, batch_id: str, taskid: str, error: str = "RuntimeError: CUDA out of memory") -> None:
        task = next(t for t in self.uploads[batch_id]["tasks"] if t["taskid"] == taskid)
        self.progress.setdefault(batch_id, []).append(
            {"taskid": taskid, "version": task["version"], "error": error, "ts": "2026-11-05T00:00:00Z"})
