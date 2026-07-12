"""Regression tests for the runtime-audit repair batch.

Each test pins one confirmed defect fixed during the agent runtime audit.
They are deliberately pure-unit (no provider, no network) so they run in
isolation regardless of environment provisioning.
"""

from __future__ import annotations

import base64

import pytest


# ── reasoning_timeouts: dash-form Anthropic sonnet ids get the floor ──────


@pytest.mark.parametrize(
    "model,expected",
    [
        # Dashed API-id form (Anthropic's own message API) — previously
        # returned None because the table only carried the dotted form.
        ("claude-sonnet-4-5", 180.0),
        ("claude-sonnet-4-6", 180.0),
        ("anthropic/claude-sonnet-4-6", 180.0),
        ("anthropic/claude-sonnet-4-5-20250929", 180.0),
        # Dotted marketing form still resolves (unchanged behavior).
        ("claude-sonnet-4.5", 180.0),
        ("claude-sonnet-4.6", 180.0),
    ],
)
def test_sonnet_reasoning_floor_matches_both_id_forms(model, expected):
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor

    assert get_reasoning_stale_timeout_floor(model) == expected


@pytest.mark.parametrize("model", ["claude-sonnet-4-2", "claude-sonnet-4.2", "claude-sonnet-4-0"])
def test_sonnet_non_reasoning_versions_still_have_no_floor(model):
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor

    assert get_reasoning_stale_timeout_floor(model) is None


# ── tool_guardrails / display: memory error:null must not crash ───────────


def test_classify_tool_failure_handles_null_memory_error():
    import json

    from agent.tool_guardrails import classify_tool_failure

    # A memory result with error present-but-null used to raise
    # TypeError ('... in None') on the "exceed the limit" check. The fix
    # guards the membership test; the call must complete without raising.
    result = json.dumps({"success": False, "error": None})
    flagged, label = classify_tool_failure("memory", result)
    assert isinstance(flagged, bool) and isinstance(label, str)


def test_display_detect_tool_failure_handles_null_memory_error():
    import json

    from agent.display import _detect_tool_failure

    # Same null-error guard on the display-side classifier. Must not raise.
    result = json.dumps({"success": False, "error": None})
    flagged, label = _detect_tool_failure("memory", result)
    assert isinstance(flagged, bool) and isinstance(label, str)


# ── context_references: binary sniff reads a bounded window ───────────────


def test_reference_token_removal_preserves_newlines():
    from agent.context_references import ContextReference, _remove_reference_tokens

    message = "Fix this:\n```\nx = 1\n```\nsee @file:foo.py"
    start = message.index("@file:foo.py")
    ref = ContextReference(
        raw="@file:foo.py", kind="file", target="foo.py",
        start=start, end=len(message),
    )
    result = _remove_reference_tokens(message, [ref])
    # Newlines and code-fence structure must survive; only the token is gone.
    assert "\n```\n" in result
    assert result.count("\n") == message[:start].count("\n")
    assert "@file:foo.py" not in result


def test_is_binary_file_reads_bounded_window(tmp_path, monkeypatch):
    from agent import context_references

    target = tmp_path / "big.log"
    target.write_text("hello world\n", encoding="utf-8")

    # read_bytes() (unbounded) must not be used for the binary sniff.
    from pathlib import Path

    def _boom(self):  # pragma: no cover - fails the test if called
        raise AssertionError("unbounded read_bytes() used for binary sniff")

    monkeypatch.setattr(Path, "read_bytes", _boom)
    assert context_references._is_binary_file(target) is False


# ── anthropic_adapter: image-only user message is preserved ───────────────


def test_image_only_user_message_is_not_replaced_with_placeholder():
    from agent.anthropic_adapter import convert_messages_to_anthropic

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    _, result = convert_messages_to_anthropic(messages)
    assert len(result) == 1
    blocks = result[0]["content"]
    assert isinstance(blocks, list)
    # The image must survive; the placeholder must NOT appear.
    assert any(b.get("type") == "image" for b in blocks)
    assert not any(
        b.get("type") == "text" and b.get("text") == "(empty message)" for b in blocks
    )


def test_whitespace_only_text_message_still_gets_placeholder():
    from agent.anthropic_adapter import convert_messages_to_anthropic

    messages = [{"role": "user", "content": [{"type": "text", "text": "   "}]}]
    _, result = convert_messages_to_anthropic(messages)
    assert result[0]["content"] == [{"type": "text", "text": "(empty message)"}]


# ── bedrock_adapter: Converse image bytes are decoded, not double-encoded ──


def test_converse_image_bytes_are_raw_not_base64_string():
    from agent.bedrock_adapter import _convert_content_to_converse

    raw = b"\x89PNG\r\n\x1a\n fake png bytes"
    b64 = base64.b64encode(raw).decode("ascii")
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    blocks = _convert_content_to_converse(content)
    image_blocks = [b for b in blocks if "image" in b]
    assert len(image_blocks) == 1
    src = image_blocks[0]["image"]["source"]["bytes"]
    # botocore expects raw bytes and base64-encodes for the wire; the adapter
    # must hand it decoded bytes, not the base64 string (which double-encodes).
    assert src == raw
    assert image_blocks[0]["image"]["format"] == "png"


def test_converse_exotic_image_format_falls_back_to_jpeg():
    from agent.bedrock_adapter import _convert_content_to_converse

    raw = b"<svg></svg>"
    b64 = base64.b64encode(raw).decode("ascii")
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/svg+xml;base64,{b64}"}},
    ]
    blocks = _convert_content_to_converse(content)
    image_blocks = [b for b in blocks if "image" in b]
    assert image_blocks[0]["image"]["format"] == "jpeg"
