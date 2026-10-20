"""Behavioral contract for the regex-only refinement routing mode."""

import math
import sys
from pathlib import Path

import pytest

WEB_ROOT = Path(__file__).resolve().parents[1]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

from system.refinement_gate import RefinementGate


@pytest.mark.parametrize(
    "text",
    [
        "请明天发货。",
        "iPhone 16 Pro is ready.",
        "SUS304 stainless steel valve.",
        "订单金额为 1,200 元。",
    ],
)
def test_regex_mode_keeps_clean_text_entities_and_arabic_amounts(text):
    decision = RefinementGate("regex").decide(text)

    assert decision.should_refine is False
    assert decision.action == "keep"
    assert decision.cleanup_signals == ()


@pytest.mark.parametrize(
    ("text", "signal"),
    [
        ("这个阀门阀门需要检查。", "repeated_phrase"),
        ("嗯，这个阀门需要检查。", "disfluency"),
        ("阀门，不对，应该是球阀。", "self_correction"),
        ("数量是十二个。", "numeric_normalization"),
    ],
)
def test_regex_mode_refines_on_cleanup_signals(text, signal):
    decision = RefinementGate("regex").decide(text)

    assert decision.should_refine is True
    assert decision.action == "refine"
    assert signal in decision.cleanup_signals


def test_regex_mode_refines_when_entity_hint_is_present():
    decision = RefinementGate("regex").decide("型号需要确认。", entity_hints=["SUS304"])

    assert decision.should_refine is True
    assert decision.action == "refine"
    assert "entity_hint_present" in decision.reasons


def test_regex_mode_does_not_force_refinement_for_final_clean_text():
    decision = RefinementGate("regex").decide("请明天发货。", is_final=True)

    assert decision.should_refine is False
    assert decision.action == "keep"


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "asr_confidence": 0.01,
            "calibrated": True,
            "covers_segment": True,
            "stable": False,
            "tail_age_ms": 0,
            "revision_ratio": 1,
            "unchanged_updates": 0,
            "cooldown_active": True,
            "same_as_last_refined": True,
            "is_final": False,
        },
        {
            "asr_confidence": math.nan,
            "calibrated": False,
            "covers_segment": False,
            "stable": True,
            "tail_age_ms": 99999,
            "revision_ratio": 0,
            "unchanged_updates": 10,
            "cooldown_active": False,
            "same_as_last_refined": False,
            "is_final": True,
        },
        {
            "asr_confidence": None,
            "calibrated": True,
            "covers_segment": True,
            "stable": False,
            "cooldown_active": True,
            "same_as_last_refined": True,
            "is_final": True,
        },
    ],
)
@pytest.mark.parametrize("text", ["iPhone 16 Pro is ready.", "嗯，这个阀门需要检查。"])
def test_regex_mode_ignores_confidence_and_streaming_metadata(text, metadata):
    decision = RefinementGate("regex").decide(text, **metadata)

    assert decision.public_dict()["asr_confidence"] is None
    assert decision.public_dict()["calibrated"] is False
    assert decision.public_dict()["covers_segment"] is False
    assert decision.should_refine is ("嗯" in text)


def test_existing_gate_modes_keep_their_basic_behavior():
    clean = "请明天发货。"

    assert RefinementGate("off").decide(clean).should_refine is True
    assert RefinementGate("conservative").decide(
        clean, asr_confidence=0.99, calibrated=True, covers_segment=True
    ).should_refine is False
    assert RefinementGate("tri_state").decide(clean, is_final=True).action == "refine"


@pytest.mark.parametrize("text", ["", "   "])
def test_regex_mode_keeps_empty_or_whitespace_text(text):
    decision = RefinementGate("regex").decide(text)

    assert decision.should_refine is False
    assert decision.action == "keep"
