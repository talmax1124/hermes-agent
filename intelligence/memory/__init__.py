"""Tiered memory for the intelligence layer.

Four tiers, each with different semantics (see :mod:`intelligence.memory.tiers`):

  * ``long_term``   — durable facts and events, retrieval-ranked.
  * ``short_term``  — a bounded, recency-weighted working buffer.
  * ``preferences`` — keyed user preferences, upserted (one value per key).
  * ``household``   — keyed household/context facts, upserted.

:class:`TieredMemory` is the standalone, unit-testable store.
:class:`IntelligenceMemoryProvider` adapts it to the runtime's
``MemoryProvider`` ABC so it can be wired in without touching Hermes Core.
"""

from __future__ import annotations

from intelligence.memory.provider import IntelligenceMemoryProvider
from intelligence.memory.store import MemoryEntry, ScoredEntry, TieredMemory
from intelligence.memory.tiers import KEYED_TIERS, TIER_WEIGHTS, Tier

__all__ = [
    "Tier",
    "TIER_WEIGHTS",
    "KEYED_TIERS",
    "MemoryEntry",
    "ScoredEntry",
    "TieredMemory",
    "IntelligenceMemoryProvider",
]
