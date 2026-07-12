"""Persona registry and selection.

:func:`get_personality` is an exact lookup by key. :func:`select_personality`
is the runtime heuristic: given whatever loose signals a caller has (an
explicit persona, a platform/surface name, the user's message), it returns
the persona that best fits, always falling back to :data:`DEFAULT_PERSONALITY`.
"""

from __future__ import annotations

from typing import Dict, Optional

from intelligence.personalities.base import Personality
from intelligence.personalities.general import GENERAL
from intelligence.personalities.kitchen import KITCHEN

PERSONALITIES: Dict[str, Personality] = {
    KITCHEN.key: KITCHEN,
    GENERAL.key: GENERAL,
}

# The safe default when nothing else matches: the all-rounder.
DEFAULT_PERSONALITY = GENERAL.key

# Surfaces that imply the Kitchen persona. A voice puck on the counter or a
# smart-display in the kitchen wants terse, spoken-style answers.
_KITCHEN_SURFACES = frozenset({
    "kitchen",
    "voice",
    "speaker",
    "smart_display",
    "display",
    "counter",
})

# Message signals that lean Kitchen. Deliberately narrow — a false positive
# here truncates a reply that wanted room, so we only match cooking intent.
_KITCHEN_KEYWORDS = (
    "recipe",
    "cook",
    "bake",
    "preheat",
    "oven",
    "simmer",
    "substitute for",
    "how long do i",
    "how much",
    "grocery",
    "shopping list",
)


def register_personality(personality: Personality) -> None:
    """Register (or replace) a persona by key. Used by tests and plugins."""
    PERSONALITIES[personality.key] = personality


def get_personality(key: Optional[str]) -> Personality:
    """Return the persona for ``key``, or the default if unknown/None."""
    if key:
        found = PERSONALITIES.get(key.strip().lower())
        if found is not None:
            return found
    return PERSONALITIES[DEFAULT_PERSONALITY]


def select_personality(
    *,
    explicit: Optional[str] = None,
    surface: Optional[str] = None,
    message: Optional[str] = None,
) -> Personality:
    """Pick a persona from available signals, most authoritative first.

    Precedence:
      1. ``explicit`` — a caller-supplied persona key wins outright.
      2. ``surface`` — the platform/device the request came from.
      3. ``message`` — cooking/shopping intent in the user's text.
      4. :data:`DEFAULT_PERSONALITY`.
    """
    if explicit:
        key = explicit.strip().lower()
        if key in PERSONALITIES:
            return PERSONALITIES[key]

    if surface and surface.strip().lower() in _KITCHEN_SURFACES:
        return KITCHEN

    if message:
        low = message.lower()
        if any(kw in low for kw in _KITCHEN_KEYWORDS):
            return KITCHEN

    return PERSONALITIES[DEFAULT_PERSONALITY]


__all__ = [
    "PERSONALITIES",
    "DEFAULT_PERSONALITY",
    "register_personality",
    "get_personality",
    "select_personality",
]
