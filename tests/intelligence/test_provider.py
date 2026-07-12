"""IntelligenceMemoryProvider: MemoryProvider contract, tools, mirroring."""

from __future__ import annotations

import json

from agent.memory_provider import MemoryProvider
from intelligence.memory.provider import IntelligenceMemoryProvider
from intelligence.memory.store import TieredMemory
from intelligence.memory.tiers import Tier


def _provider(**kw) -> IntelligenceMemoryProvider:
    return IntelligenceMemoryProvider(store=TieredMemory(), **kw)


def test_is_a_memory_provider():
    p = _provider()
    assert isinstance(p, MemoryProvider)
    assert p.name == "intelligence"
    assert p.is_available() is True


def test_tool_schemas_shape():
    p = _provider()
    names = {s["name"] for s in p.get_tool_schemas()}
    assert names == {"remember", "recall"}
    remember = next(s for s in p.get_tool_schemas() if s["name"] == "remember")
    tier_enum = remember["parameters"]["properties"]["tier"]["enum"]
    assert set(tier_enum) == {t.value for t in Tier}


def test_remember_tool_call():
    p = _provider()
    out = json.loads(
        p.handle_tool_call(
            "remember",
            {"content": "User loves garlic", "tier": "preferences", "key": "garlic"},
        )
    )
    assert out["ok"] is True
    assert out["tier"] == "preferences"
    assert p.store.entries(Tier.PREFERENCES)[0].content == "User loves garlic"


def test_remember_tool_rejects_empty():
    p = _provider()
    out = json.loads(p.handle_tool_call("remember", {"content": "   "}))
    assert out["ok"] is False


def test_recall_tool_call():
    p = _provider()
    p.store.remember("Risotto needs stirring", Tier.LONG_TERM, keywords=["risotto"])
    out = json.loads(p.handle_tool_call("recall", {"query": "risotto", "limit": 3}))
    assert out["ok"] is True
    assert out["results"]
    assert "risotto" in out["results"][0]["content"].lower()
    assert "score" in out["results"][0]


def test_unknown_tool_raises():
    p = _provider()
    try:
        p.handle_tool_call("nope", {})
    except NotImplementedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected NotImplementedError")


def test_on_memory_write_mirrors_to_tiers():
    p = _provider()
    p.on_memory_write("add", "user", "Prefers metric units")
    p.on_memory_write("add", "memory", "Project deadline is Friday")
    p.on_memory_write("remove", "memory", "should be ignored")
    assert p.store.entries(Tier.PREFERENCES)[0].content == "Prefers metric units"
    assert p.store.entries(Tier.LONG_TERM)[0].content == "Project deadline is Friday"
    # 'remove' is not mirrored.
    assert len(p.store.entries(Tier.LONG_TERM)) == 1


def test_sync_turn_captures_short_term():
    p = _provider()
    p.sync_turn("what's for dinner?", "pasta")
    st = p.store.entries(Tier.SHORT_TERM)
    assert st and st[0].content == "what's for dinner?"


def test_system_prompt_block_and_prefetch():
    p = _provider()
    assert p.system_prompt_block() == ""  # empty store
    p.store.remember("Gluten free", Tier.PREFERENCES, key="diet")
    block = p.system_prompt_block()
    assert "Gluten free" in block
    p.store.remember("Blender is broken", Tier.HOUSEHOLD, key="blender")
    assert "Blender" in p.prefetch("can I use the blender")


def test_persistence_via_store_path(tmp_path):
    path = str(tmp_path / "mem.json")
    p = IntelligenceMemoryProvider(store=TieredMemory(), store_path=path)
    p.handle_tool_call("remember", {"content": "durable", "tier": "long_term"})
    # A second provider pointed at the same path (through initialize) reloads it.
    p2 = IntelligenceMemoryProvider()
    p2.initialize("sess", hermes_home=str(tmp_path))
    # initialize builds <home>/intelligence/memory.json, which differs from
    # our explicit store_path, so p2 starts empty — verify explicit path load.
    p3 = IntelligenceMemoryProvider(store_path=path)
    p3.initialize("sess")
    assert any(e.content == "durable" for e in p3.store.all_entries())


def test_initialize_uses_hermes_home(tmp_path):
    home = tmp_path
    p = IntelligenceMemoryProvider()
    p.initialize("sess", hermes_home=str(home))
    p.handle_tool_call("remember", {"content": "home fact", "tier": "long_term"})
    # Written under <home>/intelligence/memory.json
    assert (home / "intelligence" / "memory.json").exists()
    p2 = IntelligenceMemoryProvider()
    p2.initialize("sess", hermes_home=str(home))
    assert any(e.content == "home fact" for e in p2.store.all_entries())
