"""Shared fakes for testing cluster/generate.py without torch or a GPU (loaded by conftest)."""

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
    torch.bfloat16 = "bfloat16"
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
