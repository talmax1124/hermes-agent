"""Core persona value types: :class:`StyleRules` and :class:`Personality`.

These are frozen dataclasses — a persona is a piece of data, not behavior.
The style enforcer (:mod:`intelligence.style`) reads :class:`StyleRules`;
the intelligence layer (:mod:`intelligence.layer`) reads the identity block
and default skills. Keeping them inert makes personas trivial to test,
serialize, and register at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

# Phrases no persona should ever emit. The product brief calls these out by
# name: they read as evasive, robotic, or self-referential. They are folded
# into every persona's banned list by ``StyleRules.all_banned_phrases`` so a
# new persona inherits them for free (a persona can still add its own).
UNIVERSAL_BANNED_PHRASES: Tuple[str, ...] = (
    "as an ai",
    "as an ai language model",
    "as a large language model",
    "i don't have access",
    "i do not have access",
    "i don't have the ability",
    "based on our conversation",
    "based on the conversation",
)


@dataclass(frozen=True)
class StyleRules:
    """Declarative constraints on how a persona should sound.

    Attributes
    ----------
    max_sentences:
        Soft cap on sentences in a default reply. ``None`` means unbounded.
        The enforcer flags replies over the cap and — when
        :attr:`truncate_over_cap` is set — trims them.
    prefer_brevity:
        Steers the identity block toward terseness. Informational only; the
        enforcer keys truncation off :attr:`truncate_over_cap`.
    allow_lists:
        Whether bulleted / numbered lists suit this persona. A terse persona
        that answers in one sentence should not be padding with lists.
    allow_code:
        Whether fenced code blocks are expected. When ``False`` the enforcer
        still never *counts into* code, but a persona like Kitchen has no
        business emitting it.
    truncate_over_cap:
        When ``True`` and a reply exceeds :attr:`max_sentences`, the enforcer
        returns a repaired reply trimmed to the cap (never inside code/lists).
        Off by default — truncation is lossy, so a persona opts in.
    tone:
        Free-text description used only in the identity block.
    extra_banned_phrases:
        Persona-specific additions to :data:`UNIVERSAL_BANNED_PHRASES`.
    """

    max_sentences: Optional[int] = None
    prefer_brevity: bool = False
    allow_lists: bool = True
    allow_code: bool = True
    truncate_over_cap: bool = False
    tone: str = "clear and direct"
    extra_banned_phrases: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_banned_phrases(self) -> Tuple[str, ...]:
        """Universal banned phrases plus this persona's own, de-duplicated.

        Comparison is case-insensitive downstream, so entries are normalized
        to lowercase here to keep the set stable.
        """
        seen: dict[str, None] = {}
        for phrase in (*UNIVERSAL_BANNED_PHRASES, *self.extra_banned_phrases):
            key = phrase.strip().lower()
            if key:
                seen.setdefault(key, None)
        return tuple(seen)


@dataclass(frozen=True)
class Personality:
    """A named persona: identity + style + default skill domains."""

    key: str
    display_name: str
    identity: str
    style: StyleRules
    default_skills: Tuple[str, ...] = field(default_factory=tuple)

    def system_block(self) -> str:
        """Render the persona's identity + style guidance for the system prompt.

        Deterministic and byte-stable for a given persona so it never
        threatens upstream prompt caching when injected once per session.
        """
        lines = [self.identity.strip(), ""]
        lines.append("Style:")
        if self.style.max_sentences == 1:
            lines.append("- Answer in one sentence by default.")
        elif self.style.max_sentences:
            lines.append(
                f"- Keep replies to about {self.style.max_sentences} sentences by default."
            )
        if self.style.prefer_brevity:
            lines.append("- Short, natural, human. No filler, no preamble.")
        if not self.style.allow_lists:
            lines.append("- Prefer prose over lists.")
        if not self.style.allow_code:
            lines.append("- No code blocks unless explicitly asked.")
        lines.append(f"- Tone: {self.style.tone}.")
        lines.append(
            '- Never say "as an AI", "I don\'t have access", or '
            '"based on our conversation" — use your tools instead.'
        )
        return "\n".join(lines).strip()

    def with_style(self, **overrides) -> "Personality":
        """Return a copy with individual :class:`StyleRules` fields overridden."""
        return replace(self, style=replace(self.style, **overrides))
