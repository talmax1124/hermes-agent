"""The intelligence layer: persona + memory + style, tied together.

:class:`IntelligenceLayer` is the single entry point a caller wires in. It
holds the active :class:`~intelligence.personalities.base.Personality` and a
:class:`~intelligence.memory.store.TieredMemory`, and offers two things a
turn needs:

  * :meth:`build_context` — the persona identity, style guidance, known
    preferences/household facts, and query-relevant recall, assembled into a
    system-prompt block.
  * :meth:`shape_response` — run a draft reply through the style enforcer for
    the active persona.

It imports only from within the intelligence package and (optionally) the
runtime's HERMES_HOME helper, never mutating Hermes Core.
"""

from __future__ import annotations

from typing import Optional

from intelligence.memory.store import TieredMemory
from intelligence.personalities import (
    Personality,
    get_personality,
    select_personality,
)
from intelligence.style import EnforceResult, StyleEnforcer


class IntelligenceLayer:
    """Persona-aware context building and response shaping over tiered memory."""

    def __init__(
        self,
        *,
        personality: Optional[Personality] = None,
        memory: Optional[TieredMemory] = None,
        recall_k: int = 5,
    ) -> None:
        self._personality = personality or get_personality(None)
        self._memory = memory if memory is not None else TieredMemory()
        self._enforcer = StyleEnforcer()
        self._recall_k = recall_k

    # -- Persona -------------------------------------------------------------

    @property
    def personality(self) -> Personality:
        return self._personality

    @property
    def memory(self) -> TieredMemory:
        return self._memory

    def use_personality(self, personality: Personality) -> None:
        self._personality = personality

    def route(
        self,
        *,
        explicit: Optional[str] = None,
        surface: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Personality:
        """Select and activate a persona from runtime signals; return it."""
        self._personality = select_personality(
            explicit=explicit, surface=surface, message=message
        )
        return self._personality

    # -- Context building ----------------------------------------------------

    def build_context(self, query: str = "") -> str:
        """Assemble the persona + memory system-prompt block for a turn.

        Sections, in order: persona identity & style, known user/household
        facts (stable), then query-relevant recall (volatile). Empty sections
        are omitted, so an empty store yields just the persona block.
        """
        sections = [self._personality.system_block()]

        summary = self._memory.context_summary()
        if summary:
            sections.append("What you know about the user and household:\n" + summary)

        if query:
            recall = self._memory.recall_block(query, k=self._recall_k)
            if recall:
                sections.append(recall)

        return "\n\n".join(s for s in sections if s.strip())

    # -- Response shaping ----------------------------------------------------

    def shape_response(self, text: str) -> EnforceResult:
        """Enforce the active persona's style on a draft reply."""
        return self._enforcer.enforce(text, self._personality.style)


__all__ = ["IntelligenceLayer"]
