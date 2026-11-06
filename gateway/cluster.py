"""Talking to Explorer (DESIGN.md 6).

ClusterBackend is what the service needs from the cluster. SlurmBackend does it with
short ssh and rsync calls: the login node for Slurm, the transfer node for files.
Every call starts at the gateway; nothing on the cluster calls back.
"""

from __future__ import annotations

import asyncio
import json
import math
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .config import Config

# Slurm states after which a job will never run again.
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL", "CANCELLED", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


class ClusterError(Exception):
    """The cluster could not be reached, or a command on it failed."""


@dataclass
class PollResult:
    jobs: list[tuple[str, str]] = field(default_factory=list)  # (job id, state), lowest id first
    progress: list[dict] = field(default_factory=list)  # parsed progress.jsonl lines
    claimed: bool = False  # a warm worker took over this handed-off batch
    finished: bool = False  # the worker has written every task's progress line

    @property
    def worker(self) -> tuple[str, str] | None:
        """The job that does the work: the lowest job ID with the batch's name."""
        return self.jobs[0] if self.jobs else None


@dataclass
class WarmWorker:
    """A job that finished its batch and is waiting for another (DESIGN.md 6.5)."""

    job_id: str
    idle_until: float  # Unix seconds; the worker exits then unless handed a batch
    job_end: float  # Slurm's end time for the job
    state: str  # from sacct; only RUNNING workers are handed work


class ClusterBackend(Protocol):
    async def ping(self) -> None: ...
    async def upload_batch(self, batch_id: str, payload: dict) -> None: ...
    async def submit(self, batch_id: str, n_images: int) -> str: ...
    async def poll(self, batch_id: str, job_id: str | None = None) -> PollResult: ...
    async def fetch(self, batch_id: str, dest: Path) -> None: ...
    async def cleanup(self, batch_id: str, cancel: list[str]) -> None: ...
    async def warm_workers(self) -> list[WarmWorker]: ...
    async def hand_off(self, job_id: str, batch_id: str) -> None: ...
    async def take_back(self, job_id: str) -> bool: ...


def time_limit_minutes(n_images: int, base_minutes: int = 90) -> int:
    """A fixed part plus 15 seconds per image, capped at the gpu partition's 8 hours.

    Start-up (container, packages, model) has taken from under 2 to over 11 minutes,
    depending on the node, and a warm job may serve further batches, so the fixed
    part is generous. Fair share is charged for time used, not the limit.
    """
    return min(480, base_minutes + math.ceil(n_images * 15 / 60))


def parse_sacct(text: str) -> list[tuple[str, str]]:
    jobs = {}
    for line in text.splitlines():
        job_id, _, state = line.strip().partition("|")
        if job_id.isdigit() and state:
            jobs[job_id] = state.split()[0]  # "CANCELLED by 123" -> CANCELLED; a job listed twice counts once
    return sorted(jobs.items(), key=lambda j: int(j[0]))


def parse_progress(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line still being written; it is read again next poll
        if isinstance(rec, dict) and "taskid" in rec and "version" in rec:
            records.append(rec)
    return records


_SPLIT = "---sdg-progress---"
_CLAIMED = "---sdg-claimed---"
_FINISHED = "---sdg-finished---"


class SlurmBackend:
    def __init__(self, config: Config):
        self.c = config
        self.env_dir = config.env_dir

    def _ssh_args(self, host: str) -> list[str]:
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]
        if self.c.ssh_key:
            args += ["-i", str(Path(self.c.ssh_key).expanduser()), "-o", "IdentitiesOnly=yes"]
        return args + [f"{self.c.user}@{host}"]

    async def _run(self, args: list[str], stdin: bytes | None = None, timeout: float = 120) -> str:
        proc = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise ClusterError(f"{args[0]} timed out after {timeout:.0f}s") from None
        if proc.returncode != 0:
            raise ClusterError(f"{args[0]} exited {proc.returncode}: {err.decode(errors='replace').strip()[:300]}")
        return out.decode(errors="replace")

    async def _login(self, command: str, **kw) -> str:
        return await self._run(self._ssh_args(self.c.login_host) + [command], **kw)

    async def _xfer(self, command: str, **kw) -> str:
        return await self._run(self._ssh_args(self.c.xfer_host) + [command], **kw)

    def _path(self, *parts: str) -> str:
        return shlex.quote("/".join([self.env_dir, *parts]))

    async def ping(self) -> None:
        await self._login("true", timeout=60)

    async def upload_batch(self, batch_id: str, payload: dict) -> None:
        d = self._path("batches", batch_id)
        f = self._path("batches", batch_id, "batch.json")
        tmp = self._path("batches", batch_id, "batch.json.tmp")
        await self._xfer(f"mkdir -p {d} && cat > {tmp} && mv {tmp} {f}",
                         stdin=json.dumps(payload).encode())

    async def submit(self, batch_id: str, n_images: int) -> str:
        name = shlex.quote(batch_id)
        out = await self._login(
            f"cd {self._path()} && sbatch --parsable --job-name={name} --dependency=singleton"
            f" --time={time_limit_minutes(n_images, self.c.job_minutes)}"
            f" --export=ALL,SDG_IDLE_S={int(self.c.idle_s)} sd_job.sh {name}"
        )
        job_id = out.strip().split(";")[0]
        if not job_id.isdigit():
            raise ClusterError(f"unexpected sbatch output: {out.strip()[:200]}")
        return job_id

    async def poll(self, batch_id: str, job_id: str | None = None) -> PollResult:
        """Jobs found by the batch's name, plus the known job ID.

        The name finds duplicate submissions; the ID still finds the job if Slurm
        loses its name (seen after a failed scontrol call).
        """
        name = shlex.quote(batch_id)
        user = shlex.quote(self.c.user)
        by_id = f" sacct -n -X -P -j {shlex.quote(job_id)} -o JobID,State;" if job_id else ""
        out = self._path("out", batch_id)
        text = await self._login(
            f"sacct -n -X -P -u {user} -S now-7days --name={name} -o JobID,State;{by_id}"
            f" test -f {out}/claimed && echo {_CLAIMED}; test -f {out}/finished && echo {_FINISHED};"
            f" echo {_SPLIT}; cat {out}/progress.jsonl 2>/dev/null || true"
        )
        head, _, progress_text = text.partition(_SPLIT)
        return PollResult(parse_sacct(head), parse_progress(progress_text),
                          claimed=_CLAIMED in head, finished=_FINISHED in head)

    async def warm_workers(self) -> list[WarmWorker]:
        workers = self._path("workers")
        text = await self._login(
            f"for f in {workers}/*.json; do [ -f \"$f\" ] && cat \"$f\" && echo; done; echo {_SPLIT};"
            f" ids=$(ls {workers} 2>/dev/null | sed -n 's/\\.json$//p' | paste -sd, -);"
            f' [ -n "$ids" ] && sacct -n -X -P -j "$ids" -o JobID,State || true'
        )
        files, _, states_text = text.partition(_SPLIT)
        states = dict(parse_sacct(states_text))
        found = []
        for line in files.splitlines():
            try:
                w = json.loads(line)
                found.append(WarmWorker(str(w["job_id"]), float(w["idle_until"]), float(w["job_end"]),
                                        states.get(str(w["job_id"]), "UNKNOWN")))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue  # a file being rewritten; seen again next time
        return found

    async def hand_off(self, job_id: str, batch_id: str) -> None:
        inbox = self._path("inbox")
        box = self._path("inbox", job_id)
        await self._xfer(f"mkdir -p {inbox} && printf %s {shlex.quote(batch_id)} > {box}.tmp && mv {box}.tmp {box}")

    async def take_back(self, job_id: str) -> bool:
        """Remove an unclaimed hand-off. False if the worker claimed it first (it renames the file)."""
        out = await self._xfer(f"rm {self._path('inbox', job_id)} 2>/dev/null && echo taken || echo claimed")
        return out.strip() == "taken"

    async def fetch(self, batch_id: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        src = f"{self.c.user}@{self.c.xfer_host}:{self.env_dir}/out/{batch_id}/"
        ssh = " ".join(shlex.quote(a) for a in self._ssh_args(self.c.xfer_host)[:-1])
        await self._run(["rsync", "-a", "--partial", "-e", ssh, src, f"{dest}/"], timeout=900)

    async def cleanup(self, batch_id: str, cancel: list[str]) -> None:
        """Remove the batch's files; cancel duplicate submissions by ID.

        Never by name: the job that ran the batch may still be running, warm.
        """
        if cancel:
            await self._login(f"scancel {' '.join(shlex.quote(j) for j in cancel)} || true")
        await self._xfer(f"rm -rf {self._path('batches', batch_id)} {self._path('out', batch_id)}")
