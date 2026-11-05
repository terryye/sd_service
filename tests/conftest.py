from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_cluster import FakeBackend  # noqa: E402

from gateway.app import create_app  # noqa: E402
from gateway.config import Config  # noqa: E402
from gateway.service import Service  # noqa: E402


class Clock:
    def __init__(self, t: float = 1_793_900_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def word_tokens(text: str) -> int:
    """Stands in for the CLIP tokenizer: one token per word, plus start and end."""
    return len(text.split()) + 2


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data", user="tester", env="test")


@pytest.fixture
def backend(tmp_path: Path) -> FakeBackend:
    return FakeBackend(tmp_path / "remote")


@pytest.fixture
def service(config, backend, clock) -> Service:
    return Service(config, backend, word_tokens, clock)


@pytest.fixture
def client(service):
    with TestClient(create_app(service, run_loops=False)) as c:
        yield c


def run(coro):
    return asyncio.run(coro)


def tune(service: Service, **settings) -> None:
    """Change settings of a running service (Config is frozen, so swap it)."""
    service.c = dataclasses.replace(service.c, **settings)
