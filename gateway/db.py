"""SQLite storage. The tasks table is also the queue (DESIGN.md 5).

Every state change of a task happens here, so the rules of DESIGN.md 4 live in one
place. Times are Unix seconds from the injected clock.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    taskid          TEXT PRIMARY KEY,
    prompt          TEXT NOT NULL,
    negative_prompt TEXT NOT NULL DEFAULT '',  -- the client's own; built-in terms are added at batch time
    filename_prefix TEXT NOT NULL,
    count           INTEGER NOT NULL,
    postprocess     TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    style           TEXT,               -- a configured LoRA style, or NULL for plain SDXL
    version         INTEGER NOT NULL,
    result          TEXT NOT NULL,      -- queued | running | success | needs_review | failed
    images          TEXT NOT NULL,      -- JSON list of {file, seed, result}
    error           TEXT,
    attempts        INTEGER NOT NULL,
    batch_id        TEXT,
    created         REAL NOT NULL,
    updated         REAL NOT NULL,
    queued_at       REAL NOT NULL       -- queue order; kept when a task is requeued
);
CREATE INDEX IF NOT EXISTS tasks_queue ON tasks (result, queued_at);
CREATE TABLE IF NOT EXISTS batches (
    batch_id     TEXT PRIMARY KEY,
    seq          INTEGER NOT NULL UNIQUE,
    state        TEXT NOT NULL,         -- submitting | submitted | done
    job_id       TEXT,
    job_state    TEXT,
    tasks        TEXT NOT NULL,         -- JSON list of taskids
    created      REAL NOT NULL,
    submit_tried REAL,                  -- last upload + sbatch attempt
    finished     REAL,
    handoff_at   REAL                   -- set while the batch is handed to a warm job, not sbatch'd
);
"""

DONE = ("success", "needs_review", "failed")


@dataclass
class Task:
    taskid: str
    prompt: str
    negative_prompt: str
    filename_prefix: str
    count: int
    postprocess: str
    seed: int
    style: str | None
    version: int
    result: str
    images: list[dict]
    error: str | None
    attempts: int
    batch_id: str | None
    created: float
    updated: float
    queued_at: float

    def seeds(self) -> list[int]:
        """Seed of each candidate image: the task seed, then derived ones."""
        return [(self.seed + i * 1_000_003) % 2**32 for i in range(self.count)]


@dataclass
class Batch:
    batch_id: str
    seq: int
    state: str
    job_id: str | None
    job_state: str | None
    tasks: list[str]
    created: float
    submit_tried: float | None
    finished: float | None
    handoff_at: float | None = None


def _task(row: sqlite3.Row) -> Task:
    d = dict(row)
    d["images"] = json.loads(d["images"])
    return Task(**d)


def _batch(row: sqlite3.Row) -> Batch:
    d = dict(row)
    d["tasks"] = json.loads(d["tasks"])
    return Batch(**d)


class Database:
    def __init__(self, path: Path, clock: Callable[[], float]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        # Explicit transactions. The connection is shared by the API and the background
        # loops; they all run on one event loop, so access never overlaps.
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(SCHEMA)
        # Columns added after the first release; older databases get them here.
        columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(tasks)")}
        added = (("style", "style TEXT"), ("negative_prompt", "negative_prompt TEXT NOT NULL DEFAULT ''"))
        for column, ddl in added:
            if column not in columns:
                self.conn.execute(f"ALTER TABLE tasks ADD COLUMN {ddl}")
        if "handoff_at" not in {r["name"] for r in self.conn.execute("PRAGMA table_info(batches)")}:
            self.conn.execute("ALTER TABLE batches ADD COLUMN handoff_at REAL")

    def close(self) -> None:
        self.conn.close()

    def _tx(self):
        return _Transaction(self.conn)

    # ---- tasks ----------------------------------------------------------------

    def get_task(self, taskid: str) -> Task | None:
        row = self.conn.execute("SELECT * FROM tasks WHERE taskid = ?", (taskid,)).fetchone()
        return _task(row) if row else None

    def get_tasks(self, taskids: Iterable[str]) -> dict[str, Task]:
        ids = list(taskids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(f"SELECT * FROM tasks WHERE taskid IN ({marks})", ids)
        return {r["taskid"]: _task(r) for r in rows}

    def submit(self, task: dict, on_duplicate: str) -> tuple[dict, Task | None]:
        """Insert or replace one validated task (DESIGN.md 4.2).

        Returns the API answer for it, and the old task when a finished task was
        replaced, so the caller can delete its old images.
        """
        now = self.clock()
        with self._tx():
            old = self.get_task(task["taskid"])
            if old is None:
                self.conn.execute(
                    "INSERT INTO tasks (taskid, prompt, negative_prompt, filename_prefix, count, postprocess, seed,"
                    " style, version, result, images, error, attempts, batch_id, created, updated, queued_at)"
                    " VALUES (:taskid, :prompt, :negative_prompt, :filename_prefix, :count, :postprocess, :seed,"
                    " :style, 1, 'queued', '[]', NULL, 0, NULL, :now, :now, :now)",
                    {**task, "now": now},
                )
                return {"taskid": task["taskid"], "result": "queued", "version": 1}, None
            if on_duplicate == "reject":
                return {"taskid": task["taskid"], "result": "rejected", "error": "taskid exists"}, None
            if old.result == "running":
                return {"taskid": task["taskid"], "result": "rejected", "error": "task is running"}, None
            version = old.version + 1
            # A queued task keeps its place in the queue; a finished one starts over.
            queued_at = old.queued_at if old.result == "queued" else now
            self.conn.execute(
                "UPDATE tasks SET prompt = :prompt, negative_prompt = :negative_prompt,"
                " filename_prefix = :filename_prefix, count = :count,"
                " postprocess = :postprocess, seed = :seed, style = :style, version = :version, result = 'queued',"
                " images = '[]', error = NULL, attempts = 0, batch_id = NULL, updated = :now,"
                " queued_at = :queued_at WHERE taskid = :taskid",
                {**task, "version": version, "now": now, "queued_at": queued_at},
            )
            answer = {"taskid": task["taskid"], "result": "queued", "version": version, "replaced": True}
            return answer, (old if old.result in DONE else None)

    def queued(self) -> list[Task]:
        rows = self.conn.execute("SELECT * FROM tasks WHERE result = 'queued' ORDER BY queued_at, rowid")
        return [_task(r) for r in rows]

    def complete_task(self, taskid: str, version: int, batch_id: str, images: list[dict]) -> bool:
        """Record a task's images. Ignored unless the task is still running in this batch."""
        result = "success" if any(i["result"] == "success" for i in images) else "needs_review"
        cur = self.conn.execute(
            "UPDATE tasks SET result = ?, images = ?, error = NULL, updated = ?"
            " WHERE taskid = ? AND version = ? AND batch_id = ? AND result = 'running'",
            (result, json.dumps(images), self.clock(), taskid, version, batch_id),
        )
        return cur.rowcount == 1

    def fail_task(self, taskid: str, error: str) -> None:
        self.conn.execute(
            "UPDATE tasks SET result = 'failed', error = ?, updated = ? WHERE taskid = ? AND result = 'queued'",
            (error, self.clock(), taskid),
        )

    def charge_attempt(self, taskid: str, batch_id: str, error: str, max_attempts: int) -> None:
        """A generation attempt failed: requeue the task, or fail it after max_attempts."""
        self.conn.execute(
            "UPDATE tasks SET attempts = attempts + 1,"
            " result = CASE WHEN attempts + 1 >= ? THEN 'failed' ELSE 'queued' END,"
            " error = CASE WHEN attempts + 1 >= ? THEN ? ELSE NULL END,"
            " batch_id = CASE WHEN attempts + 1 >= ? THEN batch_id ELSE NULL END,"
            " updated = ? WHERE taskid = ? AND batch_id = ? AND result = 'running'",
            (max_attempts, max_attempts, error, max_attempts, self.clock(), taskid, batch_id),
        )

    # ---- batches --------------------------------------------------------------

    def claim_batch(self, prefix: str, taskids: list[str]) -> Batch:
        """Move queued tasks into a new batch, in one transaction."""
        now = self.clock()
        with self._tx():
            seq = self.conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM batches").fetchone()[0]
            batch_id = f"{prefix}{seq:06d}"
            self.conn.execute(
                "INSERT INTO batches (batch_id, seq, state, tasks, created) VALUES (?, ?, 'submitting', ?, ?)",
                (batch_id, seq, json.dumps(taskids), now),
            )
            marks = ",".join("?" * len(taskids))
            self.conn.execute(
                f"UPDATE tasks SET result = 'running', batch_id = ?, updated = ?"
                f" WHERE taskid IN ({marks}) AND result = 'queued'",
                (batch_id, now, *taskids),
            )
        return self.get_batch(batch_id)

    def get_batch(self, batch_id: str) -> Batch:
        return _batch(self.conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone())

    def in_flight(self) -> list[Batch]:
        rows = self.conn.execute("SELECT * FROM batches WHERE state != 'done' ORDER BY seq")
        return [_batch(r) for r in rows]

    def batch_tasks(self, batch_id: str) -> list[Task]:
        rows = self.conn.execute("SELECT * FROM tasks WHERE batch_id = ? ORDER BY queued_at, rowid", (batch_id,))
        return [_task(r) for r in rows]

    def mark_submit_tried(self, batch_id: str) -> None:
        self.conn.execute("UPDATE batches SET submit_tried = ? WHERE batch_id = ?", (self.clock(), batch_id))

    def mark_submitted(self, batch_id: str, job_id: str) -> None:
        self.conn.execute(
            "UPDATE batches SET state = 'submitted', job_id = ? WHERE batch_id = ?", (job_id, batch_id)
        )

    def mark_handed_off(self, batch_id: str, job_id: str) -> None:
        self.conn.execute(
            "UPDATE batches SET state = 'submitted', job_id = ?, handoff_at = ? WHERE batch_id = ?",
            (job_id, self.clock(), batch_id),
        )

    def reset_to_submitting(self, batch_id: str) -> None:
        """A hand-off nobody claimed: back to the sbatch path, as if never submitted."""
        self.conn.execute(
            "UPDATE batches SET state = 'submitting', job_id = NULL, job_state = NULL, handoff_at = NULL,"
            " submit_tried = NULL WHERE batch_id = ?",
            (batch_id,),
        )

    def set_job_state(self, batch_id: str, job_id: str, job_state: str) -> None:
        self.conn.execute(
            "UPDATE batches SET job_id = ?, job_state = ? WHERE batch_id = ?", (job_id, job_state, batch_id)
        )

    def finish_batch(self, batch_id: str) -> None:
        self.conn.execute(
            "UPDATE batches SET state = 'done', finished = ? WHERE batch_id = ?", (self.clock(), batch_id)
        )

    # ---- health ---------------------------------------------------------------

    def queue_stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(count), 0), MIN(queued_at) FROM tasks WHERE result = 'queued'"
        ).fetchone()
        oldest = row[2]
        return {
            "queued_tasks": row[0],
            "queued_images": row[1],
            "oldest_queued_age_s": round(self.clock() - oldest) if oldest is not None else None,
        }


class _Transaction:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type, exc, tb):
        self.conn.execute("COMMIT" if exc_type is None else "ROLLBACK")
