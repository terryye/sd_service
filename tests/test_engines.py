"""Two engines: SDXL and Qwen-Image-2.1 (DESIGN.md 7)."""

import json
import sys

from conftest import run, tune
from test_queue import first_batch_done_and_warm
from worker_fakes import TASK, progress_of, run_job, write_batch

from gateway.config import Style
from gateway.service import SAFETY_NEGATIVE

VECTOR = Style(name="vector", source="civitai:1219697",
               sha256="b0024ab16d212bb1f08ccbc3b25cbe781e3aef8b00f91526778ca1b1a2c61480", trigger="color icon")


def test_qwen21_is_the_default_and_reads_long_prompts(service):
    tune(service, default_engine="qwen21")
    long_prompt = " ".join(["word"] * 300)  # far past SDXL's 77 tokens
    [a] = service.submit([{"taskid": "a", "prompt": long_prompt}])
    assert a["result"] == "queued"
    assert service.db.get_task("a").engine == "qwen21"
    [b] = service.submit([{"taskid": "b", "prompt": " ".join(["word"] * 600)}])
    assert b["error"] == "prompt too long: 602 tokens, at most 512 for qwen21"


def test_qwen21_negative_prompt_is_empty_unless_the_task_gives_one(service, backend):
    tune(service, default_engine="qwen21", max_wait_s=0)
    service.submit([{"taskid": "plain", "prompt": "a cat", "postprocess": "none"},
                    {"taskid": "neg", "prompt": "a cat", "postprocess": "none", "negative_prompt": "dogs"}])
    batch = run(service.batcher_tick())
    payload = backend.uploads[batch.batch_id]
    negatives = {t["taskid"]: t["negative_prompt"] for t in payload["tasks"]}
    assert payload["engine"] == "qwen21"
    assert negatives == {"plain": "", "neg": f"{SAFETY_NEGATIVE}, dogs"}  # guidance on only for "neg"
    assert backend.submit_engines == ["qwen21"]


def test_a_style_decides_the_engine(service):
    tune(service, default_engine="qwen21", styles={"vector": VECTOR})
    [a, b] = service.submit([{"taskid": "a", "prompt": "an apple", "style": "vector"},
                             {"taskid": "b", "prompt": "an apple", "style": "vector", "engine": "qwen21"}])
    assert a["result"] == "queued" and service.db.get_task("a").engine == "sdxl"
    assert b["error"] == "style 'vector' works only with engine sdxl"
    [c] = service.submit([{"taskid": "c", "prompt": "an apple", "engine": "dalle"}])
    assert c["error"] == "engine must be one of qwen21, sdxl"


def test_a_batch_holds_one_engine_oldest_first(service, backend, clock):
    tune(service, max_wait_s=0)
    service.submit([{"taskid": "s1", "prompt": "a", "engine": "sdxl"}])
    clock.advance(1)
    service.submit([{"taskid": "q1", "prompt": "b", "engine": "qwen21"},
                    {"taskid": "s2", "prompt": "c", "engine": "sdxl"}])
    first = run(service.batcher_tick())
    assert first.tasks == ["s1", "s2"] and first.engine == "sdxl"
    backend.set_state(first.batch_id, "RUNNING")
    for t in first.tasks:
        backend.finish_task(first.batch_id, t)
    backend.finish(first.batch_id)
    run(service.poller_tick())
    second = run(service.batcher_tick())
    assert second.tasks == ["q1"] and second.engine == "qwen21"
    assert backend.submit_engines == ["sdxl", "qwen21"]


def test_hand_off_only_to_a_warm_job_of_the_same_engine(service, backend, clock):
    first, job_id = first_batch_done_and_warm(service, backend, clock)  # an sdxl job, now warm
    service.submit([{"taskid": "q", "prompt": "a cat", "engine": "qwen21"}])
    batch = run(service.batcher_tick())
    assert backend.inbox == {} and batch.handoff_at is None  # sbatch, not the sdxl job
    assert backend.submit_engines == ["sdxl", "qwen21"]


def test_engines_endpoint(client, service):
    body = client.get("/api/engines").json()
    assert body["default"] == "sdxl"
    assert {e["name"]: e["max_prompt_tokens"] for e in body["engines"]} == {"sdxl": 77, "qwen21": 512}


# ---- the worker ---------------------------------------------------------------------------


def qwen_batch(env, batch_id, tasks):
    (env / "batches" / batch_id).mkdir(parents=True, exist_ok=True)
    (env / "batches" / batch_id / "batch.json").write_text(
        json.dumps({"batch_id": batch_id, "engine": "qwen21", "styles": {}, "tasks": tasks}))


def test_qwen21_worker_uses_guidance_only_with_a_negative_prompt(generate, tmp_path, monkeypatch):
    calls = []
    real = sys.modules["diffusers"].StableDiffusionXLPipeline._pipe

    class QwenPipe(real):
        def __call__(self, **kw):
            calls.append({k: v for k, v in kw.items() if k in ("true_cfg_scale", "negative_prompt",
                                                                "num_inference_steps")})
            return super().__call__(negative_prompt=None, guidance_scale=None, **{
                k: v for k, v in kw.items() if k in ("prompt", "generator")})

    monkeypatch.setattr(sys.modules["diffusers"], "QwenImage21Pipeline",
                        type("Q", (), {"from_pretrained": staticmethod(lambda *a, **k: QwenPipe())}), raising=False)
    env = tmp_path / "env"
    qwen_batch(env, "b1", [{**TASK, "taskid": "plain", "count": 1, "seeds": [1], "negative_prompt": ""},
                           {**TASK, "taskid": "neg", "count": 1, "seeds": [2], "negative_prompt": "dogs"}])
    run_job(generate, tmp_path, monkeypatch, "b1")
    assert calls[0] == {"num_inference_steps": 40}
    assert calls[1] == {"num_inference_steps": 40, "negative_prompt": ["dogs"], "true_cfg_scale": 4.0}
    assert {r["taskid"] for r in progress_of(env, "b1")} == {"plain", "neg"}


def test_a_worker_refuses_a_batch_for_another_engine(generate, tmp_path, monkeypatch):
    env = tmp_path / "env"
    write_batch(env, "b1", [TASK])  # an sdxl batch starts an sdxl job
    qwen_batch(env, "b2", [{**TASK, "taskid": "q"}])
    (env / "inbox").mkdir(parents=True)
    (env / "inbox" / "77").write_text("b2")
    import time
    run_job(generate, tmp_path, monkeypatch, "b1", "--job-id", "77", "--job-end", str(time.time() + 7200),
            "--idle-s", "1")
    [rec] = progress_of(env, "b2")
    assert rec["error"] == "batch for engine qwen21 reached a sdxl job"
    assert (env / "out" / "b2" / "finished").exists()
