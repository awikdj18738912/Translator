import pytest

from translator.schemas import EntityInput
from translator.terminology import TOKEN_RE, protect


def entity(canonical, *, source_aliases=(), zh_aliases=(), language="en",
           domain="general", policy="normalize", enabled=True):
    return EntityInput(
        canonical_zh=canonical,
        source_aliases=list(source_aliases),
        zh_aliases=list(zh_aliases),
        language=language,
        domain=domain,
        policy=policy,
        enabled=enabled,
    ).model_dump() | {"id": canonical}


def only_token(protected):
    tokens = TOKEN_RE.findall(protected.text)
    assert len(tokens) == 1
    return tokens[0]


def test_source_term_is_protected_and_restored_as_canonical_translation():
    terms = [entity("量子点", source_aliases=["quantum dot"])]
    protected = protect(
        "A quantum dot detector", terms, domain="general", language="en", phase="source"
    )

    token = only_token(protected)
    assert protected.text == f"A {token} detector"
    translated_candidate = f"{token}探测器"
    assert protected.restore(translated_candidate) == "量子点探测器"


def test_ascii_alias_requires_whole_word_boundaries_but_matches_punctuation():
    terms = [entity("猫", source_aliases=["cat"])]
    source = "concatenate cat_scans (CAT), cat."

    protected = protect(source, terms, domain="general", language="en", phase="source")

    assert len(TOKEN_RE.findall(protected.text)) == 2
    assert "concatenate" in protected.text
    assert "cat_scans" in protected.text
    restored = protected.restore(protected.text)
    assert restored == "concatenate cat_scans (猫), 猫."


def test_longer_overlapping_term_wins():
    terms = [
        entity("纽约", source_aliases=["York"]),
        entity("纽约市", source_aliases=["New York"]),
    ]
    protected = protect(
        "New York is large", terms, domain="general", language="en", phase="source"
    )

    assert len(TOKEN_RE.findall(protected.text)) == 1
    assert protected.restore(protected.text) == "纽约市 is large"


def test_general_domain_entry_applies_inside_selected_domain():
    terms = [entity("量子点", source_aliases=["quantum dot"])]
    protected = protect(
        "quantum dot", terms, domain="medical", language="en", phase="source"
    )

    assert protected.restore(protected.text) == "量子点"


def test_domain_specific_entry_overrides_general_for_same_span():
    terms = [
        entity("量子点", source_aliases=["quantum dot"]),
        entity("量子点探测器", source_aliases=["quantum dot"], domain="medical"),
    ]
    protected = protect(
        "quantum dot", terms, domain="medical", language="en", phase="source"
    )

    assert protected.restore(protected.text) == "量子点探测器"
    assert protected.warnings == []


def test_conflicting_alias_targets_are_left_untouched_with_warning():
    terms = [
        entity("量子点", source_aliases=["quantum dot"]),
        entity("量子微粒", source_aliases=["quantum dot"]),
    ]

    protected = protect(
        "quantum dot", terms, domain="general", language="en", phase="source"
    )

    assert protected.text == "quantum dot"
    assert protected.replacements == {}
    assert any("多个标准译名" in warning for warning in protected.warnings)


def test_source_aliases_are_filtered_by_language():
    terms = [entity("量子点", source_aliases=["quantum dot"], language="ja")]

    english = protect(
        "quantum dot", terms, domain="general", language="en", phase="source"
    )
    japanese = protect(
        "quantum dot", terms, domain="general", language="ja", phase="source"
    )

    assert english.text == "quantum dot"
    assert japanese.restore(japanese.text) == "量子点"


def test_preserve_policy_restores_observed_source_spelling():
    terms = [entity("量子点", source_aliases=["Quantum Dot"], policy="preserve")]
    protected = protect(
        "Quantum Dot", terms, domain="general", language="en", phase="source"
    )

    assert protected.restore(protected.text) == "Quantum Dot"


def test_explicit_chinese_alias_normalizes_to_canonical_spelling():
    terms = [entity("肿瘤坏死因子", zh_aliases=["TNF"])]
    protected = protect(
        "TNF受体", terms, domain="medical", language="zh", phase="zh"
    )

    assert protected.restore(protected.text) == "肿瘤坏死因子受体"


@pytest.mark.parametrize(
    "candidate",
    [
        "text without the required token",
        "{token} {token}",
        "text __TR_0123456789ab_0000__",
    ],
    ids=["lost", "duplicated", "new-token"],
)
def test_restore_rejects_lost_duplicated_or_fabricated_placeholder(candidate):
    protected = protect(
        "quantum dot", [entity("量子点", source_aliases=["quantum dot"])],
        domain="general", language="en", phase="source",
    )
    token = only_token(protected)

    with pytest.raises(ValueError, match="占位符|未恢复"):
        protected.restore(candidate.format(token=token))


def test_restore_rejects_reordered_protected_placeholders_when_order_is_required():
    terms = [
        entity("量子点", source_aliases=["quantum dot"]),
        entity("显微镜", source_aliases=["microscope"]),
    ]
    protected = protect(
        "quantum dot and microscope", terms,
        domain="general", language="en", phase="source",
    )
    tokens = TOKEN_RE.findall(protected.text)
    assert len(tokens) == 2

    with pytest.raises(ValueError, match="顺序"):
        protected.restore(" ".join(reversed(tokens)), ordered=True)


def test_chinese_refinement_protects_number_and_unit_as_one_fact():
    protected = protect(
        "剂量为3.5 kg。", [], domain="medical", language="zh", phase="zh", facts=True
    )

    token = only_token(protected)
    assert protected.text == f"剂量为{token}。"
    assert protected.restore(protected.text) == "剂量为3.5 kg。"

