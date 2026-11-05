"""LoRA styles configured in config.toml (DESIGN.md 7.1)."""

import pytest
from conftest import run, tune

from gateway.config import Style, download_url, load

ICON = Style(name="icon3d", source="hf:linoyts/3d-icon-sdxl-lora/pytorch_lora_weights.safetensors",
             sha256="4623a4b46c6335dfcf7b282922c879adf3d45ab5197de5bd79ba9efeb8fc74b5",
             trigger="3d icon in the style of <s0><s1>", scale=0.9,
             embeddings="hf:linoyts/3d-icon-sdxl-lora/embeddings.safetensors",
             embeddings_sha256="8707994599a5702845dde85d517d54cde8b10b0d35824590ea49bbe84200642a",
             embedding_tokens=("<s0>", "<s1>"))
VECTOR = Style(name="vector", source="civitai:1219697",
               sha256="b0024ab16d212bb1f08ccbc3b25cbe781e3aef8b00f91526778ca1b1a2c61480", trigger="color icon")


@pytest.fixture
def styled(service):
    tune(service, styles={"icon3d": ICON, "vector": VECTOR}, max_wait_s=0)
    return service


def test_unknown_style_is_rejected(styled):
    [a] = styled.submit([{"taskid": "a", "prompt": "a cat", "style": "anime"}])
    assert a == {"taskid": "a", "result": "rejected", "error": "unknown style 'anime' (available: icon3d, vector)"}


def test_trigger_words_count_toward_the_token_limit(styled):
    # The fake counter counts words: "color icon," adds 2 to the prompt's 73 words + 2 = 77.
    fits = " ".join(["word"] * 73)
    [a, b] = styled.submit([{"taskid": "a", "prompt": fits, "style": "vector"},
                            {"taskid": "b", "prompt": fits + " more", "style": "vector"}])
    assert a["result"] == "queued"
    assert b["error"] == "prompt too long: 78 tokens, at most 77"


def test_payload_carries_the_full_prompt_and_only_the_styles_in_use(styled, backend):
    styled.submit([{"taskid": "a", "prompt": "a red apple", "style": "icon3d"},
                   {"taskid": "b", "prompt": "a cat"}])
    batch = run(styled.batcher_tick())
    payload = backend.uploads[batch.batch_id]
    assert payload["styles"] == {"icon3d": {"lora": "icon3d-4623a4b46c63.safetensors", "scale": 0.9,
                                            "embeddings": "icon3d-8707994599a5.emb.safetensors",
                                            "tokens": ["<s0>", "<s1>"]}}
    a, b = payload["tasks"]
    assert (a["style"], a["prompt"]) == ("icon3d", "3d icon in the style of <s0><s1>, a red apple")
    assert (b["style"], b["prompt"]) == (None, "a cat")


def test_status_shows_the_style_and_keeps_the_clients_prompt(styled):
    styled.submit([{"taskid": "a", "prompt": "a red apple", "style": "vector"}])
    assert styled.task_view(styled.db.get_task("a"))["style"] == "vector"
    assert styled.db.get_task("a").prompt == "a red apple"


def test_style_removed_from_config_fails_its_queued_tasks(styled, backend):
    styled.submit([{"taskid": "a", "prompt": "a cat", "style": "vector"},
                   {"taskid": "b", "prompt": "a dog"}])
    tune(styled, styles={"icon3d": ICON})
    batch = run(styled.batcher_tick())
    assert batch.tasks == ["b"]
    task = styled.db.get_task("a")
    assert task.result == "failed" and "no longer configured" in task.error


def test_style_config_parsing_and_checks(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[cluster]\nuser = "ye.te"\n[downloads]\ncivitai_token = "abc"\n'
                    '[[styles]]\nname = "vector"\nsource = "civitai:1219697"\n'
                    f'sha256 = "{VECTOR.sha256}"\ntrigger = "color icon"\nscale = 0.8\n')
    c = load(path)
    assert c.civitai_token == "abc" and c.styles["vector"].scale == 0.8
    assert download_url("civitai:1219697") == "https://civitai.com/api/download/models/1219697"
    with pytest.raises(ValueError, match="sha256"):
        Style(name="x", source="civitai:1", sha256="nope")
    with pytest.raises(ValueError, match="source must be"):
        Style(name="x", source="https://example.com/x.safetensors", sha256=VECTOR.sha256)
    with pytest.raises(ValueError, match="embedding_tokens"):
        Style(name="x", source="civitai:1", sha256=VECTOR.sha256,
              embeddings="civitai:2", embeddings_sha256=VECTOR.sha256)
