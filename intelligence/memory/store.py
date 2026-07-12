"""The tiered memory store — standalone and unit-testable.

:class:`TieredMemory` holds entries across the four :class:`Tier` values and
answers relevance-ranked :meth:`search` queries. It has no I/O beyond
explicit :meth:`save` / :meth:`load`, and its clock is injectable, so every
scoring and decay path is deterministic under test.

Scoring (see :meth:`_score`) combines four factors:

    tier_weight × relevance × salience_factor × recency_factor

* **relevance** — token overlap between the query and the entry (explicit
  keyword hits count extra). With no query it falls back to a neutral base
  so context-prefetch still surfaces the most salient recent facts.
* **salience** — caller-supplied importance, clamped to a modest band.
* **recency** — per-tier exponential decay; short-term decays fast, keyed
  tiers effectively not at all.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from intelligence.memory.tiers import (
    KEYED_TIERS,
    SHORT_TERM_CAPACITY,
    TIER_WEIGHTS,
    Tier,
)

# Per-tier half-life (seconds) for the recency factor. Short-term is a working
# buffer that should fade within a session; long-term fades over a month;
# keyed tiers use a very long half-life so preferences/household stay strong.
_HALF_LIFE: Dict[Tier, float] = {
    Tier.SHORT_TERM: 6 * 3600.0,
    Tier.LONG_TERM: 30 * 86400.0,
    Tier.PREFERENCES: 3650 * 86400.0,
    Tier.HOUSEHOLD: 3650 * 86400.0,
}

# Minimal English stopword set — enough to stop "the"/"a"/"is" from dominating
# token overlap without pulling in a dependency.
_STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be by do for from how i if in is it its me my of on or
    our so the to us we what when where which who why will with you your""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> Set[str]:
    """Lowercase word tokens, stopwords and single chars dropped."""
    return {
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    }


@dataclass
class MemoryEntry:
    """A single stored memory."""

    id: str
    tier: Tier
    content: str
    keywords: Tuple[str, ...] = ()
    key: Optional[str] = None
    salience: float = 1.0
    source: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    # Cached token sets (not serialized) — rebuilt on construction / load.
    _tokens: Set[str] = field(default_factory=set, repr=False, compare=False)
    _keyword_tokens: Set[str] = field(default_factory=set, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.tier = Tier.from_value(self.tier)
        self._rebuild_tokens()

    def _rebuild_tokens(self) -> None:
        self._keyword_tokens = set()
        for kw in self.keywords:
            self._keyword_tokens |= _tokenize(kw)
        self._tokens = _tokenize(self.content) | self._keyword_tokens

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "tier": self.tier.value,
            "content": self.content,
            "keywords": list(self.keywords),
            "key": self.key,
            "salience": self.salience,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MemoryEntry":
        return cls(
            id=str(data["id"]),
            tier=Tier.from_value(str(data["tier"])),
            content=str(data.get("content", "")),
            keywords=tuple(data.get("keywords", []) or ()),  # type: ignore[arg-type]
            key=data.get("key"),  # type: ignore[arg-type]
            salience=float(data.get("salience", 1.0)),
            source=str(data.get("source", "")),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


@dataclass
class ScoredEntry:
    """A retrieval hit: the entry plus its computed relevance score."""

    entry: MemoryEntry
    score: float


class TieredMemory:
    """In-memory tiered store with relevance-ranked retrieval and persistence."""

    VERSION = 1

    def __init__(
        self,
        *,
        clock: Optional[Callable[[], float]] = None,
        short_term_capacity: int = SHORT_TERM_CAPACITY,
    ) -> None:
        self._clock = clock or time.time
        self._short_term_capacity = max(1, int(short_term_capacity))
        self._entries: Dict[Tier, List[MemoryEntry]] = {t: [] for t in Tier}
        self._counter = 0

    # -- Writes --------------------------------------------------------------

    def remember(
        self,
        content: str,
        tier: "str | Tier" = Tier.LONG_TERM,
        *,
        keywords: Optional[Sequence[str]] = None,
        key: Optional[str] = None,
        salience: float = 1.0,
        source: str = "",
    ) -> MemoryEntry:
        """Store an entry and return it.

        For keyed tiers (preferences, household) this upserts: an existing
        entry with the same key is replaced in place (``created_at`` preserved,
        ``updated_at`` bumped) rather than duplicated. Short-term evicts its
        oldest entry once it exceeds capacity.
        """
        tier = Tier.from_value(tier)
        content = content.strip()
        if not content:
            raise ValueError("cannot remember empty content")
        now = self._clock()
        kw = tuple(k for k in (keywords or ()) if k and k.strip())

        if tier in KEYED_TIERS:
            resolved_key = (key or content).strip().lower()
            existing = self._find_keyed(tier, resolved_key)
            if existing is not None:
                existing.content = content
                existing.keywords = kw
                existing.salience = salience
                existing.source = source or existing.source
                existing.updated_at = now
                existing._rebuild_tokens()
                return existing
            entry = self._new_entry(
                tier, content, kw, resolved_key, salience, source, now
            )
            self._entries[tier].append(entry)
            return entry

        entry = self._new_entry(tier, content, kw, key, salience, source, now)
        bucket = self._entries[tier]
        bucket.append(entry)
        if tier is Tier.SHORT_TERM and len(bucket) > self._short_term_capacity:
            # Evict oldest by updated_at (stable: list is roughly insertion order).
            bucket.sort(key=lambda e: e.updated_at)
            del bucket[: len(bucket) - self._short_term_capacity]
        return entry

    def _new_entry(
        self,
        tier: Tier,
        content: str,
        keywords: Tuple[str, ...],
        key: Optional[str],
        salience: float,
        source: str,
        now: float,
    ) -> MemoryEntry:
        self._counter += 1
        return MemoryEntry(
            id=f"{tier.value}-{self._counter}",
            tier=tier,
            content=content,
            keywords=keywords,
            key=key,
            salience=salience,
            source=source,
            created_at=now,
            updated_at=now,
        )

    def _find_keyed(self, tier: Tier, key: str) -> Optional[MemoryEntry]:
        for entry in self._entries[tier]:
            if entry.key == key:
                return entry
        return None

    def forget(self, entry_id: str) -> bool:
        """Remove an entry by id. Returns True if something was removed."""
        for bucket in self._entries.values():
            for i, entry in enumerate(bucket):
                if entry.id == entry_id:
                    del bucket[i]
                    return True
        return False

    def clear(self, tier: "str | Tier | None" = None) -> None:
        """Clear one tier, or all tiers when ``tier`` is None."""
        if tier is None:
            for bucket in self._entries.values():
                bucket.clear()
        else:
            self._entries[Tier.from_value(tier)].clear()

    # -- Reads ---------------------------------------------------------------

    def entries(self, tier: "str | Tier") -> List[MemoryEntry]:
        """Return a copy of one tier's entries (insertion order)."""
        return list(self._entries[Tier.from_value(tier)])

    def all_entries(self) -> List[MemoryEntry]:
        out: List[MemoryEntry] = []
        for tier in Tier:
            out.extend(self._entries[tier])
        return out

    def search(
        self,
        query: str,
        *,
        tiers: Optional[Iterable["str | Tier"]] = None,
        k: int = 5,
        min_score: float = 0.0,
    ) -> List[ScoredEntry]:
        """Return up to ``k`` entries ranked by relevance to ``query``.

        With an empty query, ranks by tier weight × salience × recency — the
        "what matters right now" view used for context prefetch. Ties break by
        recency (newer first) then id, so results are deterministic.
        """
        if tiers is None:
            candidate_tiers = list(Tier)
        else:
            candidate_tiers = [Tier.from_value(t) for t in tiers]

        query_tokens = _tokenize(query)
        now = self._clock()

        scored: List[ScoredEntry] = []
        for tier in candidate_tiers:
            for entry in self._entries[tier]:
                score = self._score(entry, query_tokens, now)
                if score > min_score:
                    scored.append(ScoredEntry(entry=entry, score=score))

        scored.sort(key=lambda s: (-s.score, -s.entry.updated_at, s.entry.id))
        return scored[: max(0, k)]

    def _score(self, entry: MemoryEntry, query_tokens: Set[str], now: float) -> float:
        if query_tokens:
            matched = query_tokens & entry._tokens
            if not matched:
                return 0.0
            coverage = len(matched) / len(query_tokens)
            keyword_hits = len(matched & entry._keyword_tokens)
            relevance = coverage + 0.15 * keyword_hits
        else:
            # Neutral relevance so context prefetch ranks by tier/salience/recency.
            relevance = 0.25

        salience_factor = 0.6 + 0.4 * _clamp(entry.salience, 0.0, 2.0)
        recency_factor = self._recency(entry, now)
        return TIER_WEIGHTS[entry.tier] * relevance * salience_factor * recency_factor

    def _recency(self, entry: MemoryEntry, now: float) -> float:
        age = max(0.0, now - entry.updated_at)
        half_life = _HALF_LIFE[entry.tier]
        # Range (0.5, 1.0]: even ancient entries keep half their weight, so a
        # highly-relevant old fact still beats an irrelevant fresh one.
        return 0.5 + 0.5 * math.exp(-age * math.log(2) / half_life)

    # -- Formatted views (for system prompt / recall injection) --------------

    def recall_block(self, query: str, *, k: int = 5) -> str:
        """A compact, ranked recall block for injection into a turn's context."""
        hits = self.search(query, k=k)
        if not hits:
            return ""
        lines = ["Relevant memory:"]
        for hit in hits:
            lines.append(f"- {hit.entry.content}")
        return "\n".join(lines)

    def context_summary(self) -> str:
        """Preferences + household facts, for the stable system-prompt block."""
        sections: List[str] = []
        prefs = self._entries[Tier.PREFERENCES]
        if prefs:
            sections.append("Preferences:")
            sections.extend(f"- {e.content}" for e in prefs)
        household = self._entries[Tier.HOUSEHOLD]
        if household:
            sections.append("Household:")
            sections.extend(f"- {e.content}" for e in household)
        return "\n".join(sections)

    # -- Persistence ---------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "counter": self._counter,
            "entries": [e.to_dict() for e in self.all_entries()],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, object],
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> "TieredMemory":
        store = cls(clock=clock)
        max_counter = 0
        for raw in data.get("entries", []):  # type: ignore[union-attr]
            entry = MemoryEntry.from_dict(raw)  # type: ignore[arg-type]
            store._entries[entry.tier].append(entry)
            suffix = entry.id.rsplit("-", 1)[-1]
            if suffix.isdigit():
                max_counter = max(max_counter, int(suffix))
        store._counter = max(int(data.get("counter", 0) or 0), max_counter)
        return store

    def save(self, path: "str | Path") -> None:
        """Persist to ``path`` as JSON, creating parent dirs, written atomically."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(p)

    @classmethod
    def load(
        cls,
        path: "str | Path",
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> "TieredMemory":
        """Load from ``path``; return an empty store if the file is absent."""
        p = Path(path)
        if not p.exists():
            return cls(clock=clock)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data, clock=clock)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


__all__ = ["MemoryEntry", "ScoredEntry", "TieredMemory"]
