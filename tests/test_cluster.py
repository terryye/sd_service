"""SlurmBackend command building and parsing, and dev/prod separation (DESIGN.md 6)."""

from pathlib import Path

import pytest
from conftest import run

from gateway.cluster import SlurmBackend, parse_progress, parse_sacct, time_limit_minutes
from gateway.config import Config, load


def recording(config):
    backend = SlurmBackend(config)
    calls = []

    async def fake_run(args, stdin=None, timeout=120):
        calls.append(args)
        return "4242\n" if "sbatch" in args[-1] else ""

    backend._run = fake_run
    return backend, calls


def test_parse_sacct_sorts_by_job_id_and_cleans_states():
    text = "1500|COMPLETED\n1499|CANCELLED by 12345\n\njunk\n"
    assert parse_sacct(text) == [("1499", "CANCELLED"), ("1500", "COMPLETED")]


def test_parse_progress_skips_a_half_written_line():
    text = '{"taskid": "a", "version": 1, "images": []}\n{"taskid": "b", "vers'
    assert parse_progress(text) == [{"taskid": "a", "version": 1, "images": []}]


def test_time_limit():
    assert time_limit_minutes(4) == 31
    assert time_limit_minutes(200) == 80
    assert time_limit_minutes(10_000) == 480


@pytest.mark.parametrize("env", ["dev", "prod"])
def test_commands_stay_inside_the_environment(env):
    config = Config(user="ye.te", env=env, remote_dir="~/sd_service")
    backend, calls = recording(config)
    batch_id = f"{config.batch_prefix}000007"
    run(backend.upload_batch(batch_id, {"batch_id": batch_id, "tasks": []}))
    assert run(backend.submit(batch_id, 8)) == "4242"
    run(backend.poll(batch_id))
    run(backend.fetch(batch_id, Path("/tmp/sdg-test-fetch")))
    run(backend.cleanup(batch_id))
    other = "prod" if env == "dev" else "dev"
    for args in calls:
        joined = " ".join(args)
        assert f"sd_service/{env}" in joined or f"sdg-{env}-" in joined  # its folder or its job names
        assert f"sd_service/{other}" not in joined and f"sdg-{other}-" not in joined
    assert config.batch_prefix == f"sdg-{env}-"
    submit = calls[1][-1]
    assert f"--job-name=sdg-{env}-000007" in submit and "--dependency=singleton" in submit
    assert submit.startswith(f"cd sd_service/{env} && sbatch")


def test_config_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[server]\nlisten = "0.0.0.0:9000"\ndata_dir = "d"\n'
                    '[cluster]\nuser = "ye.te"\nenv = "prod"\n[poll]\ninterval_s = 10\n')
    c = load(path)
    assert (c.host, c.port, c.data_dir, c.env, c.poll_interval_s) == ("0.0.0.0", 9000, tmp_path / "d", "prod", 10)
    path.write_text("[cluster]\nenviron = 'x'\n")
    with pytest.raises(ValueError, match="unknown key"):
        load(path)
    with pytest.raises(ValueError, match="env must be"):
        Config(env="../prod")
