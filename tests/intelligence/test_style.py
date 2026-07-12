"""Style enforcement: banned phrases, brevity, shape flags."""

from __future__ import annotations

from intelligence.personalities import GENERAL, KITCHEN
from intelligence.personalities.base import StyleRules
from intelligence.style import StyleEnforcer


def _kinds(result):
    return [v.kind for v in result.violations]


def test_clean_reply_passes_untouched():
    enf = StyleEnforcer()
    r = enf.enforce("Preheat to 180.", KITCHEN.style)
    assert r.ok
    assert not r.changed
    assert r.text == "Preheat to 180."


def test_banned_phrase_detected_and_stripped():
    enf = StyleEnforcer()
    r = enf.enforce("As an AI, I recommend butter.", KITCHEN.style)
    assert "banned_phrase" in _kinds(r)
    assert "as an ai" not in r.text.lower()
    assert "butter" in r.text
    assert r.changed


def test_longest_banned_phrase_matched_first():
    enf = StyleEnforcer()
    r = enf.enforce(
        "As an AI language model I cannot cook, but preheat to 180.",
        KITCHEN.style,
    )
    # The whole specific phrase is removed, leaving clean prose (no "language
    # model" fragment left behind by a shorter prefix match).
    assert "language model" not in r.text.lower()
    assert r.text == "I cannot cook, but preheat to 180."


def test_verbose_clean_prose_truncated_for_kitchen():
    enf = StyleEnforcer()
    r = enf.enforce(
        "Preheat to 180. Then add butter. Then wait ten minutes.",
        KITCHEN.style,
    )
    assert "too_long" in _kinds(r)
    assert r.truncated
    assert r.text == "Preheat to 180."


def test_no_truncation_when_banned_phrase_stripped():
    # Stripping already altered the text; further truncation could keep a
    # mangled fragment. We flag too_long but do not truncate.
    enf = StyleEnforcer()
    r = enf.enforce(
        "As an AI, I don't have access. But the oven is at 200C. Also more.",
        KITCHEN.style,
    )
    assert "too_long" in _kinds(r)
    assert not r.truncated
    # The informative content survives.
    assert "200C" in r.text


def test_general_persona_allows_length():
    enf = StyleEnforcer()
    text = "First point. Second point. Third point. Fourth point."
    r = enf.enforce(text, GENERAL.style)
    assert "too_long" not in _kinds(r)
    assert not r.truncated
    assert r.text == text


def test_code_block_not_counted_and_blocks_truncation():
    enf = StyleEnforcer()
    # A terse persona that happens to allow code: sentences around code should
    # not trigger truncation (has_code guard), though it is still flagged.
    style = StyleRules(max_sentences=1, truncate_over_cap=True, allow_code=True)
    text = "Run this. ```python\nprint(1)\nprint(2)\n``` Done now."
    r = enf.enforce(text, style)
    assert not r.truncated
    assert "```" in r.text  # code preserved intact


def test_list_flagged_when_disallowed():
    enf = StyleEnforcer()
    text = "Here you go:\n- one\n- two"
    r = enf.enforce(text, KITCHEN.style)
    assert "list_not_allowed" in _kinds(r)


def test_code_flagged_when_disallowed():
    enf = StyleEnforcer()
    text = "Sure:\n```py\nx=1\n```"
    r = enf.enforce(text, KITCHEN.style)
    assert "code_not_allowed" in _kinds(r)


def test_empty_input_is_ok():
    enf = StyleEnforcer()
    r = enf.enforce("", KITCHEN.style)
    assert r.ok
    assert r.text == ""
