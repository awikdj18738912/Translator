import asyncio
from dataclasses import replace
import re

import pytest

from translator.config import ModelConfig, Settings
from translator.pipeline import Pipeline, apply_patch, refinement_rejection, split_text
from translator.providers import ModelError
from translator.schemas import EntityInput
from translator.store import Store
from translator.terminology import TOKEN_RE


class FakeChat:
    def __init__(self, refine=None, translation=None):
        self.calls = []
        self.refine = refine
        self.translation = translation

    async def generate(self, config, system, user):
        self.calls.append((config.model, system, user))
        if config.model == "translation":
            if self.translation:
                return self.translation(user)
            tokens = TOKEN_RE.findall(user)
            return f"{tokens[0] if tokens else '最低起订量'}为20件，我们不能更改。"
        if isinstance(self.refine, Exception):
            raise self.refine
        return self.refine(user) if self.refine else user


def run_pipeline(tmp_path, chat=None, **changes):
    settings = replace(Settings(data_dir=tmp_path,
        translation=ModelConfig(model="translation"), refiner=ModelConfig(model="refiner")), **changes)
    store = Store(tmp_path / "test.sqlite3")
    store.upsert_entities([EntityInput(canonical_zh="最低起订量", source_aliases=["MOQ"], language="en")])
    job = store.create_job({"text":"The MOQ is 20 pieces. We cannot change it.", "language":"en", "domain":"general"}, 10)
    pipeline = Pipeline(settings, store, chat or FakeChat(), None)
    return pipeline, job, store


def test_pipeline_preserves_all_stages_and_uses_separate_models(tmp_path):
    chat = FakeChat(refine=lambda text: text.replace("我们", "我方"))
    pipeline, job, store = run_pipeline(tmp_path, chat)
    result = asyncio.run(pipeline.run(job))
    assert result["source_text"].startswith("The MOQ")
    assert result["translation"] == "最低起订量为20件，我们不能更改。"
    assert result["refined_text"] == "最低起订量为20件，我方不能更改。"
    assert result["chunks"][0]["refinement_status"] == "accepted"
    assert [c[0] for c in chat.calls] == ["translation", "refiner"]
    assert "20件" not in chat.calls[1][2]  # Quantity-unit span is protected before refinement.
    assert store.get_job(job["id"])["status"] == "completed"


@pytest.mark.parametrize("refine", [lambda text: TOKEN_RE.sub("", text), lambda text: text.replace("不", ""), ModelError("超时")])
def test_bad_refinement_falls_back_and_retains_translation(tmp_path, refine):
    pipeline, job, _ = run_pipeline(tmp_path, FakeChat(refine=refine))
    result = asyncio.run(pipeline.run(job))
    assert result["refined_text"] == result["translation"]
    assert result["chunks"][0]["refinement_status"] == "fallback"
    assert result["needs_review"] is True
    assert result["warnings"]


def test_translation_placeholder_corruption_stops_job_and_records_output(tmp_path):
    pipeline, job, store = run_pipeline(tmp_path, FakeChat(translation=lambda _: "起订数量为20件。"))
    with pytest.raises(ModelError, match="破坏"):
        asyncio.run(pipeline.run(job))
    saved = store.get_job(job["id"])
    assert saved["stage"] == "translation_rejected"
    assert saved["result"]["chunks"][0]["translation_model_output"] == "起订数量为20件。"
    assert saved["result"]["needs_review"] is True


def test_surface_number_difference_requires_review(tmp_path):
    chat = FakeChat(translation=lambda text: f"{TOKEN_RE.findall(text)[0]}为30件，我们不能更改。")
    pipeline, job, _ = run_pipeline(tmp_path, chat)
    result = asyncio.run(pipeline.run(job))
    assert result["needs_review"]
    assert any("数字写法不一致" in w for w in result["warnings"])


def test_optional_refinement_is_explicitly_disabled(tmp_path):
    chat = FakeChat()
    pipeline, job, _ = run_pipeline(tmp_path, chat, refiner_enabled=False)
    result = asyncio.run(pipeline.run(job))
    assert result["chunks"][0]["refinement_status"] == "disabled"
    assert len(chat.calls) == 1
    assert result["models"]["refiner"] is None


def test_codex_patch_protocol_and_guards(tmp_path):
    chat = FakeChat(refine=lambda _: '{"action":"replace","source":"我们","target":"我方","reason":"local_typo"}')
    pipeline, job, _ = run_pipeline(tmp_path, chat, refiner_protocol="codex_patch")
    assert "我方不能" in asyncio.run(pipeline.run(job))["refined_text"]
    with pytest.raises(ValueError):
        apply_patch("甲甲", '{"action":"replace","source":"甲","target":"乙","reason":"local_typo"}')
    with pytest.raises(ValueError):
        apply_patch("甲", '{"action":"keep","source":"甲","target":"","reason":"no_change"}')
    with pytest.raises(ValueError):
        apply_patch("__TR_0123456789ab_0000__", '{"action":"replace","source":"0123","target":"9999","reason":"local_typo"}')


def test_chunking_loses_no_source_characters():
    source = ("Please send the quotation. 请确认型号。\n" * 200) + "x" * 2000
    chunks = split_text(source, 800)
    assert "".join(chunks) == source
    assert all(0 < len(c) <= 800 for c in chunks)


@pytest.mark.parametrize("candidate", ["请交付30件。", "请不要交付20件。", "如果可以，请交付20件。", "完全不同的内容。"])
def test_refinement_guards(candidate):
    assert refinement_rejection("请交付20件。", candidate)

