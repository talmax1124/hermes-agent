"""Personality definitions, selection precedence, and rendering."""

from __future__ import annotations

from intelligence.personalities import (
    DEFAULT_PERSONALITY,
    GENERAL,
    KITCHEN,
    get_personality,
    register_personality,
    select_personality,
)
from intelligence.personalities.base import (
    UNIVERSAL_BANNED_PHRASES,
    Personality,
    StyleRules,
)


def test_get_personality_known_and_unknown():
    assert get_personality("kitchen") is KITCHEN
    assert get_personality("KITCHEN") is KITCHEN
    assert get_personality("general") is GENERAL
    # Unknown / None falls back to the default persona.
    assert get_personality(None).key == DEFAULT_PERSONALITY
    assert get_personality("nope").key == DEFAULT_PERSONALITY


def test_select_precedence_explicit_wins():
    # Explicit beats a conflicting surface and message.
    p = select_personality(explicit="kitchen", surface="cli", message="plan my sprint")
    assert p is KITCHEN
    # An unknown explicit key is ignored, falling through to the next signal.
    p = select_personality(explicit="bogus", surface="kitchen")
    assert p is KITCHEN


def test_select_by_surface_and_message():
    assert select_personality(surface="voice") is KITCHEN
    assert select_personality(surface="smart_display") is KITCHEN
    assert select_personality(message="what's a substitute for eggs?") is KITCHEN
    # A neutral request with no signals -> default (General).
    assert select_personality(message="help me debug this stack trace").key == "general"
    assert select_personality().key == DEFAULT_PERSONALITY


def test_kitchen_style_is_terse():
    assert KITCHEN.style.max_sentences == 1
    assert KITCHEN.style.prefer_brevity is True
    assert KITCHEN.style.allow_lists is False
    assert KITCHEN.style.allow_code is False
    assert KITCHEN.style.truncate_over_cap is True


def test_general_style_is_open():
    assert GENERAL.style.max_sentences is None
    assert GENERAL.style.allow_lists is True
    assert GENERAL.style.allow_code is True
    assert GENERAL.style.truncate_over_cap is False


def test_universal_banned_phrases_inherited():
    for phrase in UNIVERSAL_BANNED_PHRASES:
        assert phrase in KITCHEN.style.all_banned_phrases
        assert phrase in GENERAL.style.all_banned_phrases
    # Kitchen adds its own on top.
    assert "i'm not sure" in KITCHEN.style.all_banned_phrases


def test_all_banned_phrases_deduped_and_lowercased():
    rules = StyleRules(extra_banned_phrases=("As An AI", "custom phrase"))
    banned = rules.all_banned_phrases
    # "as an ai" appears once despite being in both universal and extra sets.
    assert banned.count("as an ai") == 1
    assert "custom phrase" in banned


def test_system_block_mentions_one_sentence_and_bans():
    block = KITCHEN.system_block()
    assert "one sentence" in block.lower()
    assert "as an ai" in block.lower()
    # General does not claim a one-sentence rule.
    assert "one sentence" not in GENERAL.system_block().lower()


def test_with_style_override_is_nondestructive():
    terse_general = GENERAL.with_style(max_sentences=2)
    assert terse_general.style.max_sentences == 2
    # Original is unchanged (frozen dataclasses).
    assert GENERAL.style.max_sentences is None
    assert terse_general.key == GENERAL.key


def test_register_personality_roundtrip():
    custom = Personality(
        key="tester",
        display_name="Tester",
        identity="You test things.",
        style=StyleRules(),
    )
    register_personality(custom)
    try:
        assert get_personality("tester") is custom
        assert select_personality(explicit="tester") is custom
    finally:
        from intelligence.personalities.registry import PERSONALITIES

        PERSONALITIES.pop("tester", None)
