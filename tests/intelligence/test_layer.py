"""IntelligenceLayer: routing, context assembly, response shaping."""

from __future__ import annotations

from intelligence import IntelligenceLayer
from intelligence.memory.store import TieredMemory
from intelligence.memory.tiers import Tier
from intelligence.personalities import GENERAL, KITCHEN


def test_default_personality_is_general():
    layer = IntelligenceLayer()
    assert layer.personality is GENERAL


def test_route_activates_persona():
    layer = IntelligenceLayer()
    assert layer.route(surface="voice") is KITCHEN
    assert layer.personality is KITCHEN
    assert layer.route(message="debug my code").key == "general"


def test_build_context_persona_only_when_memory_empty():
    layer = IntelligenceLayer(personality=KITCHEN)
    ctx = layer.build_context("anything")
    assert "kitchen companion" in ctx.lower()
    assert "Preferences:" not in ctx
    assert "Relevant memory:" not in ctx


def test_build_context_includes_memory_sections():
    mem = TieredMemory()
    mem.remember("Vegetarian", Tier.PREFERENCES, key="diet")
    mem.remember("Cast iron pan available", Tier.HOUSEHOLD, key="pan")
    mem.remember("Seared steak last week", Tier.LONG_TERM, keywords=["steak", "sear"])
    layer = IntelligenceLayer(personality=KITCHEN, memory=mem)

    ctx = layer.build_context("how do I sear steak")
    assert "Preferences:" in ctx and "Vegetarian" in ctx
    assert "Household:" in ctx and "Cast iron" in ctx
    assert "Relevant memory:" in ctx and "steak" in ctx.lower()


def test_build_context_no_recall_without_query():
    mem = TieredMemory()
    mem.remember("Seared steak last week", Tier.LONG_TERM, keywords=["steak"])
    layer = IntelligenceLayer(personality=GENERAL, memory=mem)
    ctx = layer.build_context("")  # no query -> no recall block
    assert "Relevant memory:" not in ctx


def test_shape_response_uses_active_persona():
    layer = IntelligenceLayer(personality=KITCHEN)
    r = layer.shape_response("As an AI, preheat to 180. Then wait. Then serve.")
    assert "as an ai" not in r.text.lower()
    # Kitchen truncation only fires on clean prose; here a phrase was stripped,
    # so it stays flagged but readable.
    assert any(v.kind == "too_long" for v in r.violations)

    layer.use_personality(GENERAL)
    r2 = layer.shape_response("Point one. Point two. Point three.")
    assert r2.ok
