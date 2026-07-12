"""Tiered memory: writes, tier semantics, retrieval ranking, persistence."""

from __future__ import annotations

import pytest

from intelligence.memory.store import TieredMemory
from intelligence.memory.tiers import SHORT_TERM_CAPACITY, Tier


class FakeClock:
    """Deterministic, advanceable clock for recency tests."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_remember_and_retrieve_basic():
    mem = TieredMemory()
    mem.remember("The oven runs hot", Tier.HOUSEHOLD, key="oven")
    hits = mem.search("oven temperature")
    assert hits
    assert hits[0].entry.content == "The oven runs hot"


def test_empty_content_rejected():
    mem = TieredMemory()
    with pytest.raises(ValueError):
        mem.remember("   ", Tier.LONG_TERM)


def test_keyed_tiers_upsert_not_duplicate():
    mem = TieredMemory()
    mem.remember("User likes tea", Tier.PREFERENCES, key="drink")
    mem.remember("User likes coffee", Tier.PREFERENCES, key="drink")
    prefs = mem.entries(Tier.PREFERENCES)
    assert len(prefs) == 1
    assert prefs[0].content == "User likes coffee"


def test_keyed_default_key_is_content():
    mem = TieredMemory()
    mem.remember("Peanut allergy", Tier.HOUSEHOLD)
    mem.remember("peanut allergy", Tier.HOUSEHOLD)  # same after normalization
    assert len(mem.entries(Tier.HOUSEHOLD)) == 1


def test_long_term_appends_not_upserts():
    mem = TieredMemory()
    mem.remember("Made pasta", Tier.LONG_TERM)
    mem.remember("Made pasta", Tier.LONG_TERM)
    assert len(mem.entries(Tier.LONG_TERM)) == 2


def test_short_term_capacity_evicts_oldest():
    clock = FakeClock()
    mem = TieredMemory(clock=clock, short_term_capacity=3)
    for i in range(5):
        mem.remember(f"note {i}", Tier.SHORT_TERM)
        clock.advance(1)
    st = mem.entries(Tier.SHORT_TERM)
    assert len(st) == 3
    contents = [e.content for e in st]
    assert contents == ["note 2", "note 3", "note 4"]


def test_keyword_hit_outranks_content_only_match():
    mem = TieredMemory()
    mem.remember(
        "We talked about the risotto recipe at length",
        Tier.LONG_TERM,
    )
    mem.remember("Dinner plan", Tier.LONG_TERM, keywords=["risotto"])
    hits = mem.search("risotto", k=2)
    # Both match "risotto", but the explicit keyword entry gets the bonus.
    assert hits[0].entry.content == "Dinner plan"


def test_tier_weight_prefers_preferences():
    mem = TieredMemory()
    mem.remember("vegetarian diet", Tier.LONG_TERM)
    mem.remember("vegetarian diet", Tier.PREFERENCES, key="diet")
    hits = mem.search("vegetarian", k=2)
    assert hits[0].entry.tier is Tier.PREFERENCES


def test_recency_breaks_ties_in_short_term():
    clock = FakeClock()
    mem = TieredMemory(clock=clock)
    mem.remember("bought milk today", Tier.SHORT_TERM)
    clock.advance(10 * 3600)  # 10 hours later
    mem.remember("bought milk today", Tier.SHORT_TERM)
    hits = mem.search("milk", k=2)
    # Newer short-term entry scores higher via recency.
    assert hits[0].entry.updated_at > hits[1].entry.updated_at


def test_no_match_returns_nothing():
    mem = TieredMemory()
    mem.remember("The sky is blue", Tier.LONG_TERM)
    assert mem.search("quantum chromodynamics") == []


def test_empty_query_ranks_by_tier_and_salience():
    mem = TieredMemory()
    mem.remember("low importance note", Tier.LONG_TERM, salience=0.2)
    mem.remember("a preference", Tier.PREFERENCES, key="p", salience=1.0)
    hits = mem.search("", k=5)
    assert hits[0].entry.tier is Tier.PREFERENCES


def test_forget_and_clear():
    mem = TieredMemory()
    e = mem.remember("temporary", Tier.LONG_TERM)
    assert mem.forget(e.id) is True
    assert mem.forget(e.id) is False
    mem.remember("a", Tier.LONG_TERM)
    mem.remember("b", Tier.SHORT_TERM)
    mem.clear(Tier.LONG_TERM)
    assert mem.entries(Tier.LONG_TERM) == []
    assert mem.entries(Tier.SHORT_TERM)  # untouched
    mem.clear()
    assert mem.all_entries() == []


def test_recall_block_and_context_summary_formatting():
    mem = TieredMemory()
    mem.remember("User is vegetarian", Tier.PREFERENCES, key="diet")
    mem.remember("Toddler in the house", Tier.HOUSEHOLD, key="kids")
    mem.remember("Loves spicy food", Tier.LONG_TERM, keywords=["spicy"])
    summary = mem.context_summary()
    assert "Preferences:" in summary and "vegetarian" in summary
    assert "Household:" in summary and "Toddler" in summary
    block = mem.recall_block("spicy dinner")
    assert block.startswith("Relevant memory:")
    assert "spicy" in block.lower()
    assert mem.recall_block("nonexistent topic") == ""


def test_persistence_roundtrip(tmp_path):
    clock = FakeClock()
    mem = TieredMemory(clock=clock)
    mem.remember("durable fact", Tier.LONG_TERM, keywords=["fact"])
    mem.remember("pref value", Tier.PREFERENCES, key="k")
    path = tmp_path / "mem.json"
    mem.save(path)

    loaded = TieredMemory.load(path, clock=clock)
    assert len(loaded.all_entries()) == 2
    # Tokens rebuilt on load -> search still works.
    assert loaded.search("fact")[0].entry.content == "durable fact"
    # Keyed upsert still works after load (counter preserved, no id clash).
    loaded.remember("new fact", Tier.LONG_TERM)
    ids = [e.id for e in loaded.all_entries()]
    assert len(ids) == len(set(ids))  # unique ids


def test_load_missing_file_returns_empty(tmp_path):
    mem = TieredMemory.load(tmp_path / "does_not_exist.json")
    assert mem.all_entries() == []


def test_default_short_term_capacity_constant():
    # Guard: the module constant is what an unconfigured store uses.
    mem = TieredMemory()
    for i in range(SHORT_TERM_CAPACITY + 5):
        mem.remember(f"n{i}", Tier.SHORT_TERM)
    assert len(mem.entries(Tier.SHORT_TERM)) == SHORT_TERM_CAPACITY
