"""cluster/generate.py against the job contract (DESIGN.md 6.1), with fake torch and diffusers."""

from __future__ import annotations

import sys
import time

from PIL import Image
from worker_fakes import CLUSTER, TASK, confetti, progress_of, run_batch, run_job, sticker, wallpaper, write_batch


def test_writes_images_and_one_progress_line_per_task(generate, tmp_path, monkeypatch, capsys):
    out, progress = run_batch(generate, tmp_path, monkeypatch, [TASK])
    assert progress[0]["taskid"] == "apple" and progress[0]["version"] == 2
    assert [(i["file"], i["seed"], i["result"]) for i in progress[0]["images"]] == [
        ("app_1.webp", 11, "success"), ("app_2.webp", 12, "success")]
    for name in ("app_1.webp", "app_2.webp"):
        with Image.open(out / "apple" / "v2" / name) as im:
            assert im.format == "WEBP" and im.size == (1024, 1024)
            assert im.convert("RGB").getpixel((5, 5)) == (255, 255, 255)  # background whitened
    assert generate.calls == [[11, 12]]  # both candidates in one pass
    log = capsys.readouterr().out
    for step in ("imports done", "VAE loaded", "pipeline loaded", "moved to GPU", "apple: 2 image(s)"):
        assert step in log


def test_pattern_is_retried_with_a_new_seed(generate, tmp_path, monkeypatch):
    generate.bad_seeds.add(12)
    _, progress = run_batch(generate, tmp_path, monkeypatch, [TASK])
    second = progress[0]["images"][1]
    assert second["result"] == "success" and second["attempts"] == 2
    assert second["seed"] == generate.retry_seed(12, 1)


def test_pattern_on_every_try_needs_review(generate, tmp_path, monkeypatch):
    generate.bad_seeds.update({12} | {generate.retry_seed(12, a) for a in range(1, 4)})
    out, progress = run_batch(generate, tmp_path, monkeypatch, [TASK])
    assert progress[0]["images"][1]["result"] == "needs_review"
    assert progress[0]["images"][1]["attempts"] == 4
    with Image.open(out / "apple" / "v2" / "app_2.webp") as im:
        assert im.convert("RGB").getpixel((5, 5)) != (255, 255, 255)  # kept as drawn, not whitened


def test_rerun_skips_finished_tasks(generate, tmp_path, monkeypatch):
    run_batch(generate, tmp_path, monkeypatch, [TASK])
    generate.calls.clear()
    _, progress = run_batch(generate, tmp_path, monkeypatch, [TASK])
    assert generate.calls == [] and len(progress) == 1


def test_checks_reject_patterns_and_blank_results():
    sys.path.insert(0, str(CLUSTER))
    from postprocess import needs_retry, whiten_background

    for draw, retry in ((sticker, False), (wallpaper, True), (confetti, True)):
        _, stats = whiten_background(draw(0))
        assert needs_retry(stats) is retry, (draw.__name__, stats)


def test_styles_load_once_and_switch_per_task(generate, tmp_path, monkeypatch, capsys):
    styles = {"icon3d": {"lora": "icon3d-aaa.safetensors", "scale": 0.9,
                         "embeddings": "icon3d-bbb.emb.safetensors", "tokens": ["<s0>", "<s1>"]},
              "unused": {"lora": "unused.safetensors", "scale": 1.0, "embeddings": None, "tokens": []}}
    tasks = [{**TASK, "taskid": "styled", "count": 1, "seeds": [1], "style": "icon3d"},
             {**TASK, "taskid": "plain", "count": 1, "seeds": [2], "style": None}]
    run_batch(generate, tmp_path, monkeypatch, tasks, styles)
    assert generate.lora == [
        ("load", ["text_encoder.layer.lora.weight", "unet.down.lora.weight"], "icon3d-aaa"),  # prefix fixed
        ("embed", "L", ("<s0>", "<s1>"), "clip_l side"),
        ("embed", "G", ("<s0>", "<s1>"), "clip_g side"),
        ("enable",), ("set", ("icon3d-aaa",), (0.9,)),   # task "styled"; adapters are named after the file
        ("disable",),                                    # task "plain"
    ]
    assert "style icon3d loaded" in capsys.readouterr().out


def test_no_styles_means_no_lora_calls(generate, tmp_path, monkeypatch):
    run_batch(generate, tmp_path, monkeypatch, [TASK])
    assert generate.lora == []


def test_a_style_that_fails_to_load_fails_only_its_tasks(generate, tmp_path, monkeypatch):
    styles = {"broken": {"lora": "broken.safetensors", "scale": 1.0, "embeddings": None, "tokens": []}}
    tasks = [{**TASK, "taskid": "styled", "count": 1, "seeds": [1], "style": "broken"},
             {**TASK, "taskid": "plain", "count": 1, "seeds": [2], "style": None}]
    out, progress = run_batch(generate, tmp_path, monkeypatch, tasks, styles)
    by_task = {r["taskid"]: r for r in progress}
    assert by_task["styled"]["error"].startswith("style broken failed to load: ValueError")
    assert by_task["plain"]["images"][0]["result"] == "success"
    assert (out / "plain" / "v2" / "app_1.webp").exists()


def test_fix_prefixes(generate):
    fixed = generate.fix_prefixes({"unet.unet.a": 1, "unet.b": 2, "text_encoder_2.text_encoder_2.c": 3})
    assert fixed == {"unet.a": 1, "unet.b": 2, "text_encoder_2.c": 3}


def test_batch_ends_with_a_finished_marker(generate, tmp_path, monkeypatch):
    out, _ = run_batch(generate, tmp_path, monkeypatch, [TASK])
    assert (out / "finished").exists()


def test_warm_worker_runs_a_handed_off_batch_without_reloading(generate, tmp_path, monkeypatch):
    env = tmp_path / "env"
    write_batch(env, "b1", [{**TASK, "taskid": "first"}])
    write_batch(env, "b2", [{**TASK, "taskid": "second"}])
    (env / "inbox").mkdir(parents=True)
    (env / "inbox" / "77").write_text("b2")  # the gateway's hand-off, waiting when b1 ends
    loads = []
    sdxl = generate.diffusers.StableDiffusionXLPipeline
    monkeypatch.setattr(sdxl, "from_pretrained", lambda *a, **k: loads.append(1) or sdxl._pipe())
    # Short idle window, far-off job end: b2 is claimed at once, then the window closes.
    run_job(generate, tmp_path, monkeypatch, "b1", "--job-id", "77", "--job-end", str(time.time() + 7200),
            "--idle-s", "1")
    assert len(loads) == 1  # the model loaded once for both batches
    assert [r["taskid"] for r in progress_of(env, "b2")] == ["second"]
    assert (env / "out" / "b2" / "claimed").exists() and (env / "out" / "b2" / "finished").exists()
    assert not (env / "inbox" / "77").exists() and not (env / "inbox" / "77.claimed").exists()
    assert list((env / "workers").iterdir()) == []  # the announcement is gone once it exits


def test_no_idle_window_near_the_job_time_limit(generate, tmp_path, monkeypatch):
    env = tmp_path / "env"
    write_batch(env, "b1", [TASK])
    (env / "inbox").mkdir(parents=True)
    (env / "inbox" / "77").write_text("b2")
    run_job(generate, tmp_path, monkeypatch, "b1", "--job-id", "77", "--job-end", str(time.time() + 300),
            "--idle-s", "300")
    assert (env / "inbox" / "77").exists()  # never claimed: less than 10 minutes were left
