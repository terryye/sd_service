"""cluster/generate.py against the job contract (DESIGN.md 6.1), with fake torch and diffusers."""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

CLUSTER = Path(__file__).resolve().parent.parent / "cluster"


def sticker(seed: int) -> Image.Image:
    """One red square on a grey background, like an SDXL sticker before whitening."""
    a = np.full((1024, 1024, 3), 200, np.uint8)
    a[312:712, 312:712] = (200, 30, 30)
    return Image.fromarray(a)


def wallpaper(seed: int) -> Image.Image:
    """Many tiles: the repeating pattern the checks must reject."""
    a = np.full((1024, 1024, 3), 200, np.uint8)
    for y in range(30, 900, 200):
        for x in range(30, 900, 200):
            a[y:y + 130, x:x + 130] = (30, 120, 200)
    return Image.fromarray(a)


def confetti(seed: int) -> Image.Image:
    """Hundreds of tiny dots: cleanup removes them all and leaves a blank image."""
    a = np.full((1024, 1024, 3), 200, np.uint8)
    for y in range(40, 1000, 60):
        for x in range(40, 1000, 60):
            a[y:y + 20, x:x + 20] = (30, 120, 200)
    return Image.fromarray(a)


@pytest.fixture
def generate(monkeypatch):
    """Import generate.py with fake GPU libraries. bad_seeds decide which seeds draw a wallpaper."""
    calls, bad_seeds, lora = [], set(), []

    class Gen:
        def __init__(self, device):
            self.seed = None

        def manual_seed(self, s):
            self.seed = s
            return self

    class Pipe:
        text_encoder = tokenizer = "clip_l side"
        text_encoder_2 = tokenizer_2 = "clip_g side"

        def to(self, device):
            return self

        def load_lora_weights(self, state_dict, adapter_name):
            if "broken" in adapter_name:
                raise ValueError("Target modules not found in the base model")
            lora.append(("load", sorted(state_dict), adapter_name))

        def load_textual_inversion(self, tensor, token, text_encoder, tokenizer):
            lora.append(("embed", tensor, tuple(token), text_encoder))

        def enable_lora(self):
            lora.append(("enable",))

        def disable_lora(self):
            lora.append(("disable",))

        def set_adapters(self, names, adapter_weights):
            lora.append(("set", tuple(names), tuple(adapter_weights)))

        def set_progress_bar_config(self, **kw):
            pass

        def __call__(self, prompt, negative_prompt, generator, **kw):
            seeds = [g.seed for g in generator]
            calls.append(seeds)
            return types.SimpleNamespace(images=[wallpaper(s) if s in bad_seeds else sticker(s) for s in seeds])

    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.Generator = Gen
    diffusers = types.ModuleType("diffusers")
    diffusers.AutoencoderKL = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())
    diffusers.StableDiffusionXLPipeline = types.SimpleNamespace(from_pretrained=lambda *a, **k: Pipe(), _pipe=Pipe)
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda path: (
        {"clip_l": "L", "clip_g": "G"} if path.endswith(".emb.safetensors")
        else {"unet.unet.down.lora.weight": 1, "text_encoder.layer.lora.weight": 2})
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.syspath_prepend(str(CLUSTER))
    sys.modules.pop("generate", None)
    import generate as module

    module.calls, module.bad_seeds, module.lora = calls, bad_seeds, lora
    yield module
    sys.modules.pop("generate", None)


def write_batch(env: Path, batch_id: str, tasks, styles=None) -> None:
    (env / "batches" / batch_id).mkdir(parents=True, exist_ok=True)
    (env / "batches" / batch_id / "batch.json").write_text(
        json.dumps({"batch_id": batch_id, "styles": styles or {}, "tasks": tasks}))


def progress_of(env: Path, batch_id: str) -> list[dict]:
    return [json.loads(line) for line in (env / "out" / batch_id / "progress.jsonl").read_text().splitlines()]


def run_job(generate, tmp_path, monkeypatch, batch_id, *extra):
    monkeypatch.setattr(sys, "argv", ["generate.py", "--env", str(tmp_path / "env"), "--batch-id", batch_id,
                                      "--loras", str(tmp_path / "loras"), *extra])
    assert generate.main() == 0


def run_batch(generate, tmp_path, monkeypatch, tasks, styles=None):
    env = tmp_path / "env"
    write_batch(env, "sdg-test-000001", tasks, styles)
    run_job(generate, tmp_path, monkeypatch, "sdg-test-000001")
    return env / "out" / "sdg-test-000001", progress_of(env, "sdg-test-000001")


TASK = {"taskid": "apple", "version": 2, "prompt": "a red apple", "negative_prompt": "scary", "count": 2,
        "seeds": [11, 12], "postprocess": "sticker", "filename_prefix": "app"}


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
    monkeypatch.setattr(generate.StableDiffusionXLPipeline, "from_pretrained",
                        lambda *a, **k: loads.append(1) or generate.StableDiffusionXLPipeline._pipe())
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
