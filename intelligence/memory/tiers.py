"""Memory tiers and their retrieval semantics.

The tier an entry lives in decides three things:

  * its base weight in retrieval ranking (:data:`TIER_WEIGHTS`),
  * whether it is *keyed* — one value per key, upserted rather than appended
    (:data:`KEYED_TIERS`), and
  * whether it decays / is capacity-bounded (short-term only).

These are policy constants, deliberately separated from the store mechanics
in :mod:`intelligence.memory.store` so the policy is easy to read and test.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet, Mapping


class Tier(str, Enum):
    """The four memory tiers. ``str`` mixin so values serialize as plain text."""

    LONG_TERM = "long_term"
    SHORT_TERM = "short_term"
    PREFERENCES = "preferences"
    HOUSEHOLD = "household"

    @classmethod
    def from_value(cls, value: "str | Tier") -> "Tier":
        """Coerce a string or :class:`Tier` to a :class:`Tier` (raises on bad)."""
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


# Base multiplier applied to an entry's relevance score during retrieval.
# Preferences and household facts are load-bearing for personalization, so
# they outrank a merely-relevant long-term note; short-term is transient and
# leans on its recency bonus rather than a high base weight.
TIER_WEIGHTS: Mapping[Tier, float] = {
    Tier.PREFERENCES: 1.5,
    Tier.HOUSEHOLD: 1.4,
    Tier.LONG_TERM: 1.0,
    Tier.SHORT_TERM: 0.9,
}

# Tiers that hold at most one entry per key: writing the same key again
# replaces the prior value instead of appending a duplicate.
KEYED_TIERS: FrozenSet[Tier] = frozenset({Tier.PREFERENCES, Tier.HOUSEHOLD})

# Short-term is a bounded working buffer; the oldest entries are evicted once
# it exceeds this many items. Kept small — short-term is "this conversation",
# not history.
SHORT_TERM_CAPACITY: int = 40

__all__ = [
    "Tier",
    "TIER_WEIGHTS",
    "KEYED_TIERS",
    "SHORT_TERM_CAPACITY",
]
