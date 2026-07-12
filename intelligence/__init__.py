"""Hermes intelligence layer.

A product layer that sits *on top of* the Hermes agent runtime (the
foundation) without modifying Hermes Core. It supplies:

  * personalities — Kitchen (terse, human) and General (planning / dev /
    research) personas with declarative style rules.
  * memory        — a tiered long/short-term + preferences + household
    store with retrieval scoring, plus a ``MemoryProvider`` adapter that
    plugs into the runtime's existing provider system.
  * style         — a response-style enforcer (banned phrases, brevity,
    one-sentence default) driven by the active persona's rules.

Everything here is import-only against ``agent/`` — it never mutates it.
The :class:`IntelligenceLayer` in :mod:`intelligence.layer` ties the three
together and is the single entry point callers wire in.
"""

from __future__ import annotations

from intelligence.layer import IntelligenceLayer
from intelligence.personalities import (
    GENERAL,
    KITCHEN,
    Personality,
    StyleRules,
    get_personality,
    select_personality,
)
from intelligence.style import EnforceResult, StyleEnforcer, StyleViolation

__all__ = [
    "IntelligenceLayer",
    "Personality",
    "StyleRules",
    "KITCHEN",
    "GENERAL",
    "get_personality",
    "select_personality",
    "StyleEnforcer",
    "EnforceResult",
    "StyleViolation",
]
