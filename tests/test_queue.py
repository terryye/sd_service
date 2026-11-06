"""Batcher and poller: batching, collecting, attempts, submit failures, restarts (DESIGN.md 5-6)."""

from conftest import run, tune

from gateway.service import SAFETY_NEGATIVE, STICKER_NEGATIVE, Service


def add(service, taskid, count=1, prompt="a cat"):
    [answer] = service.submit([{"taskid": taskid, "prompt": prompt, "count": count}])
    assert answer["result"] == "queued", answer
    return answer


# ---- batching ------------------------------------------------------------------------


def test_small_batch_waits_for_the_time_limit(service, backend, clock):
    add(service, "a")
    assert run(service.batcher_tick()) is None
    clock.advance(59)
    assert run(service.batcher_tick()) is None
    clock.advance(1)
    batch = run(service.batcher_tick())
    assert batch.state == "submitted" and batch.tasks == ["a"]
    assert backend.submits == [batch.batch_id]
    assert batch.batch_id == "sdg-test-000001"


def test_full_batch_goes_at_once_and_tasks_are_never_split(service, backend):
    for i in range(51):
        add(service, f"t{i:02d}", count=4)  # 204 images queued
    batch = run(service.batcher_tick())
    assert len(batch.tasks) == 50  # 200 images; the 51st task waits for the next batch
    assert batch.tasks == [f"t{i:02d}" for i in range(50)]  # first in, first out
    assert sum(t["count"] for t in backend.uploads[batch.batch_id]["tasks"]) == 200
    assert service.db.get_task("t50").result == "queued"


def test_one_batch_in_flight_at_a_time(service, backend, clock):
    tune(service, max_wait_s=0)
    add(service, "a")
    first = run(service.batcher_tick())
    add(service, "b")
    assert run(service.batcher_tick()) is None
    backend.finish_task(first.batch_id, "a")
    backend.set_state(first.batch_id, "COMPLETED")
    run(service.poller_tick())
    second = run(service.batcher_tick())
    assert second.tasks == ["b"] and second.batch_id == "sdg-test-000002"


def test_batch_payload_matches_the_job_contract(service, backend):
    tune(service, max_wait_s=0)
    service.submit([{"taskid": "a", "prompt": "a cat", "count": 3, "seed": 7, "filename_prefix": "cat",
                     "postprocess": "none"}])
    batch = run(service.batcher_tick())
    assert backend.uploads[batch.batch_id] == {
        "batch_id": batch.batch_id, "engine": "sdxl", "styles": {}, "tasks": [
        {"taskid": "a", "version": 1, "prompt": "a cat", "count": 3, "style": None,
         "negative_prompt": SAFETY_NEGATIVE,  # postprocess none: safety terms only
         "seeds": [7, 7 + 1_000_003, 7 + 2_000_006], "postprocess": "none", "filename_prefix": "cat"}]}


# ---- collecting ------------------------------------------------------------------------


def test_results_are_collected_while_the_batch_runs(service, backend):
    tune(service, max_wait_s=0)
    add(service, "a", count=2)
    add(service, "b")
    batch = run(service.batcher_tick())
    backend.set_state(batch.batch_id, "RUNNING")
    backend.finish_task(batch.batch_id, "a")
    run(service.poller_tick())
    assert service.db.get_task("a").result == "success"
    assert service.db.get_task("b").result == "running"
    assert len(service.db.in_flight()) == 1
    backend.finish_task(batch.batch_id, "b")
    backend.set_state(batch.batch_id, "COMPLETED")
    run(service.poller_tick())
    assert service.db.get_task("b").result == "success"
    assert service.db.in_flight() == [] and backend.cleaned == [batch.batch_id]


def test_broken_image_is_not_collected(service, backend):
    tune(service, max_wait_s=0)
    add(service, "a")
    batch = run(service.batcher_tick())
    backend.set_state(batch.batch_id, "RUNNING")
    backend.finish_task(batch.batch_id, "a", size=(512, 512))
    run(service.poller_tick())
    assert service.db.get_task("a").result == "running"  # tried again next poll
    assert not (service.c.results_dir / "a").exists()


# ---- attempts --------------------------------------------------------------------------


def test_waiting_in_the_slurm_queue_never_counts(service, backend, clock):
    tune(service, max_wait_s=0)
    add(service, "a")
    batch = run(service.batcher_tick())
    for _ in range(100):  # 100 polls, 50 minutes, still PENDING
        clock.advance(30)
        run(service.poller_tick())
    task = service.db.get_task("a")
    assert task.result == "running" and task.attempts == 0
    assert backend.submits == [batch.batch_id]


def test_failed_job_requeues_unfinished_tasks_with_an_attempt(service, backend):
    tune(service, max_wait_s=0)
    add(service, "a")
    add(service, "b")
    batch = run(service.batcher_tick())
    backend.set_state(batch.batch_id, "RUNNING")
    backend.finish_task(batch.batch_id, "a")
    run(service.poller_tick())
    backend.set_state(batch.batch_id, "TIMEOUT")
    run(service.poller_tick())
    a, b = service.db.get_task("a"), service.db.get_task("b")
    assert a.result == "success" and a.attempts == 0
    assert b.result == "queued" and b.attempts == 1 and b.batch_id is None


def test_task_fails_after_three_attempts(service, backend):
    tune(service, max_wait_s=0)
    add(service, "a")
    for _ in range(3):
        batch = run(service.batcher_tick())
        backend.set_state(batch.batch_id, "NODE_FAIL")
        run(service.poller_tick())
    task = service.db.get_task("a")
    assert task.result == "failed" and task.attempts == 3
    assert "NODE_FAIL" in task.error


def test_generation_error_charges_only_that_task(service, backend):
    tune(service, max_wait_s=0)
    add(service, "bad")
    add(service, "good")
    batch = run(service.batcher_tick())
    backend.set_state(batch.batch_id, "RUNNING")
    backend.fail_task(batch.batch_id, "bad")
    backend.finish_task(batch.batch_id, "good")
    run(service.poller_tick())
    bad = service.db.get_task("bad")
    assert bad.result == "queued" and bad.attempts == 1
    assert service.db.get_task("good").result == "success"


def test_completed_job_without_a_task_requeues_it(service, backend):
    tune(service, max_wait_s=0)
    add(service, "a")
    batch = run(service.batcher_tick())
    backend.set_state(batch.batch_id, "COMPLETED")
    run(service.poller_tick())
    task = service.db.get_task("a")
    assert task.result == "queued" and task.attempts == 1


# ---- submit failures -------------------------------------------------------------------


def test_failed_sbatch_that_still_created_the_job_is_not_resubmitted(service, backend, clock):
    tune(service, max_wait_s=0)
    backend.submit_fails = [True]
    backend.ghost = True
    add(service, "a")
    batch = run(service.batcher_tick())
    assert batch.state == "submitting"
    clock.advance(30)
    run(service.poller_tick())  # sacct finds the job under the batch name
    batch = service.db.get_batch(batch.batch_id)
    assert batch.state == "submitted" and batch.job_id == "1001"
    clock.advance(600)
    run(service.poller_tick())
    assert backend.submits == [batch.batch_id]  # submitted exactly once
    assert service.db.get_task("a").attempts == 0


def test_failed_sbatch_is_resubmitted_after_five_minutes(service, backend, clock):
    tune(service, max_wait_s=0)
    backend.submit_fails = [True]
    add(service, "a")
    batch = run(service.batcher_tick())
    clock.advance(30)
    run(service.poller_tick())
    assert backend.submits == [batch.batch_id]  # too early to resubmit
    clock.advance(300)
    run(service.poller_tick())
    assert backend.submits == [batch.batch_id, batch.batch_id]
    assert service.db.get_batch(batch.batch_id).state == "submitted"
    assert service.db.get_task("a").attempts == 0


def test_lowest_job_id_is_the_worker(service, backend):
    tune(service, max_wait_s=0)
    add(service, "a")
    batch = run(service.batcher_tick())
    backend.jobs[batch.batch_id].append(("1500", "COMPLETED"))  # a duplicate that exited at once
    run(service.poller_tick())
    assert service.db.in_flight()[0].job_state == "PENDING"


def test_cluster_outage_changes_nothing(service, backend, clock):
    tune(service, max_wait_s=0)
    add(service, "a")
    batch = run(service.batcher_tick())
    backend.down = True
    for _ in range(20):
        clock.advance(30)
        run(service.poller_tick())
    assert service.db.get_task("a").result == "running"
    backend.down = False
    backend.finish_task(batch.batch_id, "a")
    backend.set_state(batch.batch_id, "COMPLETED")
    run(service.poller_tick())
    assert service.db.get_task("a").result == "success"


# ---- restarts --------------------------------------------------------------------------


def test_restart_resumes_the_in_flight_batch(service, config, backend, clock):
    from conftest import word_tokens

    tune(service, max_wait_s=0)
    add(service, "a")
    add(service, "b")
    batch = run(service.batcher_tick())
    service.db.close()

    restarted = Service(config, backend, word_tokens, clock)
    assert [b.batch_id for b in restarted.db.in_flight()] == [batch.batch_id]
    backend.finish_task(batch.batch_id, "a")
    backend.finish_task(batch.batch_id, "b")
    backend.set_state(batch.batch_id, "COMPLETED")
    run(restarted.poller_tick())
    assert {restarted.db.get_task(t).result for t in "ab"} == {"success"}
    assert backend.submits == [batch.batch_id]


def test_job_that_lost_its_name_is_still_found_by_id(service, backend):
    tune(service, max_wait_s=0)
    add(service, "a")
    batch = run(service.batcher_tick())
    job_id = backend.jobs[batch.batch_id][0][0]
    backend.jobs[batch.batch_id] = []           # sacct --name no longer finds it
    backend.nameless[job_id] = [(job_id, "FAILED")]
    run(service.poller_tick())
    assert backend.polled_with[-1] == job_id
    assert service.db.in_flight() == []
    assert service.db.get_task("a").result == "queued"  # requeued, not stuck


def test_negative_prompt_adds_to_the_built_in_terms(service, backend):
    tune(service, max_wait_s=0)
    service.submit([{"taskid": "a", "prompt": "a cat", "negative_prompt": " dogs, grass "},
                    {"taskid": "b", "prompt": "a cat", "postprocess": "none"}])
    batch = run(service.batcher_tick())
    negatives = {t["taskid"]: t["negative_prompt"] for t in backend.uploads[batch.batch_id]["tasks"]}
    assert negatives == {"a": f"{SAFETY_NEGATIVE}, {STICKER_NEGATIVE}, dogs, grass", "b": SAFETY_NEGATIVE}
    assert service.task_view(service.db.get_task("a"))["negative_prompt"] == "dogs, grass"


def test_negative_prompt_counts_the_built_in_terms(service):
    # The fake counter counts words: the sticker built-ins are 29 words + 2 = 31 "tokens", so 46 words fit.
    fits = " ".join(["x"] * 46)
    [a, b, c] = service.submit([{"taskid": "a", "prompt": "a cat", "negative_prompt": fits},
                                {"taskid": "b", "prompt": "a cat", "negative_prompt": fits + " y"},
                                {"taskid": "c", "prompt": "a cat", "negative_prompt": 5}])
    assert a["result"] == "queued"
    assert b["error"] == ("negative_prompt too long: 78 tokens with the built-in terms, at most 77 for sdxl"
                          " (about 46 left for yours)")
    assert c["error"] == "negative_prompt must be a string"


# ---- warm workers (DESIGN.md 6.5) ----------------------------------------------------


def first_batch_done_and_warm(service, backend, clock):
    """Run one sbatch'd batch to its finished marker, leaving its job RUNNING and warm."""
    tune(service, max_wait_s=0)
    add(service, "first")
    batch = run(service.batcher_tick())
    job_id = backend.jobs[batch.batch_id][0][0]
    backend.set_state(batch.batch_id, "RUNNING")
    backend.finish_task(batch.batch_id, "first")
    backend.finish(batch.batch_id)
    backend.go_warm(job_id, idle_until=clock() + 300, job_end=clock() + 5000)
    run(service.poller_tick())
    return batch, job_id


def test_finished_marker_completes_a_batch_while_the_job_stays_warm(service, backend, clock):
    batch, job_id = first_batch_done_and_warm(service, backend, clock)
    assert service.db.in_flight() == [] and service.db.get_task("first").result == "success"
    assert backend.cancelled == []  # the warm job itself is never cancelled


def test_next_batch_goes_to_the_warm_job_not_sbatch(service, backend, clock):
    first, job_id = first_batch_done_and_warm(service, backend, clock)
    add(service, "second")
    second = run(service.batcher_tick())
    assert backend.submits == [first.batch_id]  # no new sbatch
    assert backend.inbox == {job_id: second.batch_id}
    assert second.job_id == job_id and second.handoff_at is not None
    backend.claim(job_id)
    backend.finish_task(second.batch_id, "second")
    backend.finish(second.batch_id)
    run(service.poller_tick())
    assert service.db.get_task("second").result == "success"
    assert service.db.in_flight() == []


def test_unclaimed_hand_off_is_taken_back_and_sbatched(service, backend, clock):
    first, job_id = first_batch_done_and_warm(service, backend, clock)
    add(service, "second")
    second = run(service.batcher_tick())
    clock.advance(30)
    run(service.poller_tick())
    assert backend.inbox == {job_id: second.batch_id}  # still within the 90 s window
    clock.advance(70)
    run(service.poller_tick())  # taken back: back to the sbatch path, not an attempt
    assert backend.inbox == {} and service.db.get_batch(second.batch_id).state == "submitting"
    run(service.poller_tick())
    assert backend.submits == [first.batch_id, second.batch_id]
    assert service.db.get_task("second").attempts == 0


def test_hand_off_that_never_arrived_falls_back_to_sbatch(service, backend, clock):
    first, job_id = first_batch_done_and_warm(service, backend, clock)
    backend.hand_off_fails = True  # the inbox file was never written
    add(service, "second")
    second = run(service.batcher_tick())
    assert second.handoff_at is not None and backend.inbox == {}
    clock.advance(100)
    run(service.poller_tick())  # take_back finds nothing: maybe claimed, so wait
    assert service.db.get_batch(second.batch_id).state == "submitted"
    clock.advance(100)
    run(service.poller_tick())  # twice the timeout and still no claim: it never arrived
    assert service.db.get_batch(second.batch_id).state == "submitting"


def test_warm_job_too_close_to_its_time_limit_is_not_used(service, backend, clock):
    first, job_id = first_batch_done_and_warm(service, backend, clock)
    backend.go_warm(job_id, idle_until=clock() + 300, job_end=clock() + 400)
    add(service, "second")
    second = run(service.batcher_tick())
    assert backend.inbox == {} and second.handoff_at is None
    assert backend.submits == [first.batch_id, second.batch_id]


def test_idle_window_off_means_sbatch_every_time(service, backend, clock):
    first, job_id = first_batch_done_and_warm(service, backend, clock)
    tune(service, idle_s=0)
    add(service, "second")
    run(service.batcher_tick())
    assert backend.inbox == {} and len(backend.submits) == 2


def test_health_lists_warm_workers(client, service, backend, clock):
    first_batch_done_and_warm(service, backend, clock)
    run(service.poller_tick())  # nothing in flight: the heartbeat reads the warm workers
    assert client.get("/api/health").json()["warm_workers"] == [
        {"job_id": "1001", "engine": "sdxl", "idle_left_s": 300}]


def test_with_a_warm_job_a_batch_goes_after_seconds_not_a_minute(service, backend, clock):
    first, job_id = first_batch_done_and_warm(service, backend, clock)
    run(service.poller_tick())  # the idle heartbeat sees the warm job
    tune(service, max_wait_s=60)
    add(service, "second")
    assert run(service.batcher_tick()) is None  # just queued
    clock.advance(5)
    second = run(service.batcher_tick())
    assert second is not None and backend.inbox == {job_id: second.batch_id}


def test_without_a_warm_job_the_full_wait_applies(service, backend, clock):
    tune(service, max_wait_s=60)
    add(service, "a")
    clock.advance(5)
    assert run(service.batcher_tick()) is None
