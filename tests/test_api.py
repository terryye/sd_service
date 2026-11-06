"""The HTTP API: validation, duplicates, status, files, health (DESIGN.md 3 and 4)."""

from conftest import run, tune


def submit(client, *tasks):
    res = client.post("/api/images/gen", json={"tasks": list(tasks)})
    assert res.status_code == 202, res.text
    return res.json()["tasks"]


def finish(service, backend, *taskids, results=None):
    """Batch every queued task, let the fake job finish the given ones, and collect."""
    batch = run(service.batcher_tick())
    for taskid in taskids:
        backend.finish_task(batch.batch_id, taskid, results)
    backend.set_state(batch.batch_id, "COMPLETED")
    run(service.poller_tick())
    return batch


def test_submit_queues_and_reports_status(client):
    [answer] = submit(client, {"taskid": "apple-01", "prompt": "a red apple", "count": 2})
    assert answer == {"taskid": "apple-01", "result": "queued", "version": 1}
    view = client.get("/api/images/tasks/apple-01").json()
    assert view["result"] == "queued" and view["images"] == [] and view["version"] == 1
    assert view["created"].endswith("Z")


def test_each_task_is_validated_on_its_own(client):
    answers = submit(
        client,
        {"taskid": "ok-1", "prompt": "a cat"},
        {"taskid": "bad id!", "prompt": "a cat"},
        {"taskid": "t2", "prompt": "  "},
        {"taskid": "t3", "prompt": "a cat", "count": 5},
        {"taskid": "t4", "prompt": "a cat", "count": True},
        {"taskid": "t5", "prompt": "a cat", "filename_prefix": "../etc"},
        {"taskid": "t6", "prompt": "a cat", "on_duplicate": "overwrite"},
        {"taskid": "t7", "prompt": "a cat", "postprocess": "fancy"},
        {"taskid": "t8", "prompt": "a cat", "seed": -1},
        {"taskid": "t9", "prompt": "a cat", "colour": "red"},
        "not an object",
    )
    assert [a["result"] for a in answers] == ["queued"] + ["rejected"] * 10
    assert "unknown field: colour" in answers[9]["error"]
    assert answers[10] == {"taskid": None, "result": "rejected", "error": "task must be an object"}


def test_prompt_over_77_tokens_is_rejected(client):
    ok = " ".join(["word"] * 75)       # 75 words + start/end = 77 tokens
    too_long = " ".join(["word"] * 76)
    a, b = submit(client, {"taskid": "ok", "prompt": ok}, {"taskid": "long", "prompt": too_long})
    assert a["result"] == "queued"
    assert b == {"taskid": "long", "result": "rejected", "error": "prompt too long: 78 tokens, at most 77 for sdxl"}


def test_whole_request_errors(client):
    assert client.post("/api/images/gen", content=b"not json").status_code == 400
    assert client.post("/api/images/gen", json={"tasks": []}).status_code == 400
    assert client.post("/api/images/gen", json=[{"taskid": "a"}]).status_code == 400
    many = [{"taskid": f"t{i}", "prompt": "a cat"} for i in range(101)]
    assert client.post("/api/images/gen", json={"tasks": many}).status_code == 400
    assert client.get("/api/images/tasks/nope").status_code == 404


def test_duplicate_reject_is_the_default(client):
    submit(client, {"taskid": "a", "prompt": "a cat"})
    [answer] = submit(client, {"taskid": "a", "prompt": "a dog"})
    assert answer == {"taskid": "a", "result": "rejected", "error": "taskid exists"}


def test_replace_queued_task_keeps_its_place(client, service, clock):
    submit(client, {"taskid": "first", "prompt": "a cat"})
    clock.advance(1)
    submit(client, {"taskid": "second", "prompt": "a dog"})
    [answer] = submit(client, {"taskid": "first", "prompt": "a lion", "on_duplicate": "replace"})
    assert answer == {"taskid": "first", "result": "queued", "version": 2, "replaced": True}
    assert [t.taskid for t in service.db.queued()] == ["first", "second"]
    assert service.db.get_task("first").prompt == "a lion"


def test_replace_running_task_is_rejected(client, service):
    submit(client, {"taskid": "a", "prompt": "a cat", "count": 4})
    tune(service, max_wait_s=0)
    run(service.batcher_tick())
    [answer] = submit(client, {"taskid": "a", "prompt": "a dog", "on_duplicate": "replace"})
    assert answer == {"taskid": "a", "result": "rejected", "error": "task is running"}


def test_replace_finished_task_deletes_old_images(client, service, backend):
    tune(service, max_wait_s=0)
    submit(client, {"taskid": "a", "prompt": "a cat", "count": 2, "filename_prefix": "cat"})
    finish(service, backend, "a")
    assert sorted(p.name for p in (service.c.results_dir / "a").iterdir()) == ["cat_1.webp", "cat_2.webp"]
    [answer] = submit(client, {"taskid": "a", "prompt": "a dog", "count": 1, "on_duplicate": "replace"})
    assert answer["version"] == 2
    assert not (service.c.results_dir / "a").exists()
    view = client.get("/api/images/tasks/a").json()
    assert view["result"] == "queued" and view["images"] == []


def test_finished_task_serves_its_images(client, service, backend):
    tune(service, max_wait_s=0)
    submit(client, {"taskid": "a", "prompt": "a cat", "count": 2, "filename_prefix": "cat", "seed": 42})
    finish(service, backend, "a", results=["success", "needs_review"])
    view = client.get("/api/images/tasks/a").json()
    assert view["result"] == "success"
    assert view["images"] == [
        {"url": "/results/a/cat_1.webp?v=1", "seed": 42, "result": "success"},
        {"url": "/results/a/cat_2.webp?v=1", "seed": (42 + 1_000_003) % 2**32, "result": "needs_review"},
    ]
    res = client.get(view["images"][0]["url"])
    assert res.status_code == 200 and res.content[:4] == b"RIFF"


def test_all_candidates_need_review(client, service, backend):
    tune(service, max_wait_s=0)
    submit(client, {"taskid": "a", "prompt": "a cat", "count": 2})
    finish(service, backend, "a", results=["needs_review", "needs_review"])
    assert client.get("/api/images/tasks/a").json()["result"] == "needs_review"


def test_bulk_status_keeps_order_and_reports_unknown(client):
    submit(client, {"taskid": "a", "prompt": "a cat"}, {"taskid": "b", "prompt": "a dog"})
    body = client.get("/api/images/tasks?ids=b,zzz,a").json()
    assert [(t["taskid"], t["result"]) for t in body["tasks"]] == [
        ("b", "queued"), ("zzz", "not_found"), ("a", "queued")]
    assert client.get("/api/images/tasks?ids=").status_code == 400
    ids = ",".join(f"t{i}" for i in range(101))
    assert client.get(f"/api/images/tasks?ids={ids}").status_code == 400


def test_health(client, service, backend, clock):
    submit(client, {"taskid": "a", "prompt": "a cat", "count": 3})
    clock.advance(10)
    h = client.get("/api/health").json()
    assert h["cluster"] == "unknown" and h["queued_tasks"] == 1 and h["queued_images"] == 3
    assert h["oldest_queued_age_s"] == 10 and h["batches_in_flight"] == 0
    backend.down = True
    tune(service, max_wait_s=0)
    run(service.batcher_tick())  # submit fails: not an attempt, batch waits to be looked up
    assert client.get("/api/health").json()["batches_in_flight"] == 1
    clock.advance(301)
    run(service.poller_tick())
    h = client.get("/api/health").json()
    assert h["cluster"] == "unreachable" and h["status"] == "degraded"


def test_test_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "sd_service test page" in res.text
