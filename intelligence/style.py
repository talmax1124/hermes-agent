"""Response-style enforcement.

:class:`StyleEnforcer` reads a persona's :class:`~intelligence.personalities.base.StyleRules`
and both *reports* and *repairs* style problems in a draft reply:

  * **banned phrases** — the universal "as an AI" / "I don't have access" /
    "based on our conversation" set, plus any persona additions, are detected
    and stripped (a safe, meaning-preserving edit).
  * **length** — replies over ``max_sentences`` are flagged, and trimmed when
    the persona opts into ``truncate_over_cap`` (never inside code).
  * **shape** — lists or code blocks are flagged for personas that disallow
    them (reported only; never silently rewritten).

Detection is always on; repairs are conservative and never touch fenced code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from intelligence.personalities.base import StyleRules

# Matches a fenced code block (``` ... ```), non-greedy, across lines.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# Sentence boundary: a terminator followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# A list item line: bullet or numbered.
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", re.MULTILINE)


@dataclass(frozen=True)
class StyleViolation:
    """One thing wrong with a draft reply."""

    kind: str  # "banned_phrase" | "too_long" | "list_not_allowed" | "code_not_allowed"
    detail: str


@dataclass
class EnforceResult:
    """Outcome of enforcing style on a draft."""

    text: str
    violations: List[StyleViolation] = field(default_factory=list)
    truncated: bool = False

    @property
    def ok(self) -> bool:
        """True when the draft had no violations at all."""
        return not self.violations

    @property
    def changed(self) -> bool:
        """True when :attr:`text` differs from the input (a repair happened)."""
        return self._changed

    _changed: bool = False


def _strip_code(text: str) -> str:
    return _CODE_FENCE_RE.sub(" ", text)


def _split_sentences(text: str) -> List[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def _cleanup(text: str) -> str:
    """Tidy prose after a phrase removal: stray punctuation, spacing, casing."""
    # Drop a dangling leading comma/space left by removing a sentence opener.
    text = re.sub(r"^[\s,;:.\-]+", "", text)
    # Collapse whitespace runs and fix " ," artifacts.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove a leftover comma directly after a sentence start ("^, I can" -> "I can").
    text = re.sub(r"(^|[.!?]\s+),\s*", r"\1", text)
    text = text.strip()
    # Recapitalize the first alphabetical character.
    for i, ch in enumerate(text):
        if ch.isalpha():
            text = text[:i] + ch.upper() + text[i + 1 :]
            break
    return text


class StyleEnforcer:
    """Apply a persona's :class:`StyleRules` to a draft reply."""

    def enforce(self, text: str, style: StyleRules) -> EnforceResult:
        original = text or ""
        working = original
        violations: List[StyleViolation] = []

        # 1. Banned phrases — detect and strip. Longest first, so a specific
        # phrase ("as an ai language model") is removed whole before a prefix
        # ("as an ai") can bite a fragment out of it.
        stripped_any = False
        for phrase in sorted(style.all_banned_phrases, key=len, reverse=True):
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.search(working):
                violations.append(StyleViolation("banned_phrase", phrase))
                working = pattern.sub("", working)
                stripped_any = True
        if stripped_any:
            working = _cleanup(working)

        # 2. Length — count sentences in prose (code excluded), trim if opted in.
        #    Truncation is only safe on otherwise-clean prose: trimming text we
        #    just had to strip a banned phrase out of risks keeping a mangled
        #    fragment as the sole surviving sentence. In that case we flag
        #    "too_long" but leave repair to a re-prompt.
        truncated = False
        if style.max_sentences is not None:
            prose = _strip_code(working)
            sentences = _split_sentences(prose)
            has_code = bool(_CODE_FENCE_RE.search(working))
            if len(sentences) > style.max_sentences:
                violations.append(
                    StyleViolation(
                        "too_long",
                        f"{len(sentences)} sentences > {style.max_sentences}",
                    )
                )
                if style.truncate_over_cap and not has_code and not stripped_any:
                    working = " ".join(sentences[: style.max_sentences]).strip()
                    truncated = True

        # 3. Shape — report (never auto-rewrite) list/code use the persona bans.
        if not style.allow_lists and _LIST_LINE_RE.search(working):
            violations.append(StyleViolation("list_not_allowed", "reply uses a list"))
        if not style.allow_code and _CODE_FENCE_RE.search(working):
            violations.append(
                StyleViolation("code_not_allowed", "reply uses a code block")
            )

        final = working.strip()
        result = EnforceResult(text=final, violations=violations, truncated=truncated)
        result._changed = final != original.strip()
        return result


__all__ = ["StyleEnforcer", "EnforceResult", "StyleViolation"]
