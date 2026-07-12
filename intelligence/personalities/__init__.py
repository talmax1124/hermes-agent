"""Persona definitions for the intelligence layer.

A :class:`Personality` bundles an identity block (for the system prompt),
declarative :class:`StyleRules` (consumed by the style enforcer), and the
default skill domains the persona leans on. Two ship built in:

  * :data:`KITCHEN` — short, natural, human; one sentence by default.
  * :data:`GENERAL` — the all-rounder for planning, developer work, and
    research; allows length, lists, and code.

Look them up by key with :func:`get_personality`, or pick one from loose
runtime context with :func:`select_personality`.
"""

from __future__ import annotations

from intelligence.personalities.base import Personality, StyleRules
from intelligence.personalities.general import GENERAL
from intelligence.personalities.kitchen import KITCHEN
from intelligence.personalities.registry import (
    DEFAULT_PERSONALITY,
    PERSONALITIES,
    get_personality,
    register_personality,
    select_personality,
)

__all__ = [
    "Personality",
    "StyleRules",
    "KITCHEN",
    "GENERAL",
    "PERSONALITIES",
    "DEFAULT_PERSONALITY",
    "get_personality",
    "register_personality",
    "select_personality",
]
