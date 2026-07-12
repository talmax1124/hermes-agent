"""Adapt :class:`TieredMemory` to the runtime's ``MemoryProvider`` ABC.

This is the only file in the memory package that imports from ``agent/``.
It lets the intelligence layer's tiered store plug into the existing
``MemoryManager`` as an external provider — exposing ``remember`` / ``recall``
tools, injecting preferences/household into the system prompt, prefetching
ranked recall each turn, and mirroring built-in memory writes into long-term
— all without modifying Hermes Core.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from intelligence.memory.store import TieredMemory
from intelligence.memory.tiers import Tier

logger = logging.getLogger(__name__)


class IntelligenceMemoryProvider(MemoryProvider):
    """External memory provider backed by the tiered store.

    Persistence lives under ``<hermes_home>/intelligence/memory.json`` by
    default; pass ``store_path`` to override (tests use an in-memory store by
    passing a pre-built ``TieredMemory`` and no path).
    """

    def __init__(
        self,
        *,
        store: Optional[TieredMemory] = None,
        store_path: Optional[str] = None,
        recall_k: int = 5,
    ) -> None:
        self._store = store if store is not None else TieredMemory()
        self._store_path: Optional[Path] = Path(store_path) if store_path else None
        self._recall_k = recall_k
        self._prefetched: str = ""
        self._session_id: str = ""

    @property
    def name(self) -> str:
        return "intelligence"

    @property
    def store(self) -> TieredMemory:
        return self._store

    # -- Lifecycle -----------------------------------------------------------

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        if self._store_path is None:
            hermes_home = kwargs.get("hermes_home")
            if hermes_home:
                self._store_path = Path(hermes_home) / "intelligence" / "memory.json"
        if self._store_path is not None and self._store_path.exists():
            try:
                self._store = TieredMemory.load(self._store_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("intelligence memory load failed: %s", exc)

    def _persist(self) -> None:
        if self._store_path is not None:
            try:
                self._store.save(self._store_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("intelligence memory save failed: %s", exc)

    def shutdown(self) -> None:
        self._persist()

    # -- Context injection ---------------------------------------------------

    def system_prompt_block(self) -> str:
        summary = self._store.context_summary()
        if not summary:
            return ""
        return "What you know about the user and household:\n" + summary

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self._prefetched = self._store.recall_block(query, k=self._recall_k)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # Prefer a freshly-scored recall for this exact query; fall back to
        # whatever was queued after the previous turn.
        block = self._store.recall_block(query, k=self._recall_k)
        return block or self._prefetched

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # Capture the user's turn into short-term working memory so the next
        # turn can recall it. Keep it lightweight — durable facts come through
        # the explicit ``remember`` tool or ``on_memory_write`` mirroring.
        text = (user_content or "").strip()
        if text:
            self._store.remember(text, Tier.SHORT_TERM, source="turn")
            self._persist()

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Mirror built-in memory writes into long-term (or preferences when the
        # built-in write targeted the user profile).
        if action not in ("add", "replace") or not content.strip():
            return
        tier = Tier.PREFERENCES if target == "user" else Tier.LONG_TERM
        self._store.remember(content, tier, source="builtin_mirror")
        self._persist()

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        tier_values = [t.value for t in Tier]
        return [
            {
                "name": "remember",
                "description": (
                    "Save a durable fact to memory. Use for preferences, "
                    "household context, and things worth recalling later."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The fact to remember, in one line.",
                        },
                        "tier": {
                            "type": "string",
                            "enum": tier_values,
                            "description": (
                                "Which memory tier: 'preferences' for user "
                                "likes/dislikes, 'household' for home context, "
                                "'long_term' for durable facts, 'short_term' "
                                "for this-session notes."
                            ),
                        },
                        "key": {
                            "type": "string",
                            "description": (
                                "Optional stable key for preferences/household "
                                "so re-saving updates rather than duplicates."
                            ),
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "recall",
                "description": "Search memory for facts relevant to a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search memory for.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "remember":
            content = str(args.get("content", "")).strip()
            if not content:
                return json.dumps({"ok": False, "error": "content is required"})
            tier = args.get("tier") or Tier.LONG_TERM.value
            entry = self._store.remember(
                content,
                Tier.from_value(tier),
                key=args.get("key"),
                source="tool",
            )
            self._persist()
            return json.dumps({"ok": True, "id": entry.id, "tier": entry.tier.value})

        if tool_name == "recall":
            query = str(args.get("query", "")).strip()
            limit = int(args.get("limit", self._recall_k) or self._recall_k)
            hits = self._store.search(query, k=limit)
            return json.dumps({
                "ok": True,
                "results": [
                    {
                        "content": h.entry.content,
                        "tier": h.entry.tier.value,
                        "score": round(h.score, 4),
                    }
                    for h in hits
                ],
            })

        raise NotImplementedError(
            f"Provider {self.name} does not handle tool {tool_name}"
        )


__all__ = ["IntelligenceMemoryProvider"]
