"""cluster/generate.py against the job contract (DESIGN.md 6.1), with fake torch and diffusers."""

from __future__ import annotations

import json
import sys
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

        def load_lora_weights(self, path, adapter_name):
            lora.append(("load", Path(path).name, adapter_name))

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
    diffusers.StableDiffusionXLPipeline = types.SimpleNamespace(from_pretrained=lambda *a, **k: Pipe())
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda path: {"clip_l": "L", "clip_g": "G"}
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.syspath_prepend(str(CLUSTER))
    sys.modules.pop("generate", None)
    import generate as module

    module.calls, module.bad_seeds, module.lora = calls, bad_seeds, lora
    yield module
    sys.modules.pop("generate", None)


def run_batch(generate, tmp_path, monkeypatch, tasks, styles=None):
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps({"batch_id": "sdg-test-000001", "styles": styles or {}, "tasks": tasks}))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["generate.py", "--batch", str(batch), "--out", str(out),
                                      "--loras", str(tmp_path / "loras")])
    assert generate.main() == 0
    lines = (out / "progress.jsonl").read_text().splitlines()
    return out, [json.loads(line) for line in lines]


TASK = {"taskid": "apple", "version": 2, "prompt": "a red apple", "count": 2,
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
    _, progress = run_batch(generate, tmp_path, monkeypatch, [TASK])
    assert progress[0]["images"][1]["result"] == "needs_review"
    assert progress[0]["images"][1]["attempts"] == 4


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
        ("load", "icon3d-aaa.safetensors", "icon3d"),
        ("embed", "L", ("<s0>", "<s1>"), "clip_l side"),
        ("embed", "G", ("<s0>", "<s1>"), "clip_g side"),
        ("enable",), ("set", ("icon3d",), (0.9,)),   # task "styled"
        ("disable",),                                # task "plain"
    ]
    assert "style icon3d loaded" in capsys.readouterr().out


def test_no_styles_means_no_lora_calls(generate, tmp_path, monkeypatch):
    run_batch(generate, tmp_path, monkeypatch, [TASK])
    assert generate.lora == []
