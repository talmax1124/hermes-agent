"""Regression coverage for the final provider request boundary."""

from agent.message_sanitization import sanitize_provider_request_kwargs


def test_next_user_turn_drops_empty_tool_calls_at_final_provider_boundary():
    persisted = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "done", "tool_calls": []},
        {"role": "user", "content": "next"},
    ]

    final_payload = sanitize_provider_request_kwargs({"messages": persisted})

    assert all(message.get("tool_calls") != [] for message in final_payload["messages"])
    assert "tool_calls" not in final_payload["messages"][1]
    assert persisted[1]["tool_calls"] == []
    assert final_payload["messages"][1] is not persisted[1]


def test_final_provider_payload_after_mcp_tool_call_preserves_pair_and_repairs_orphan():
    """Model the wire history produced by a completed MCP tool dispatch."""
    persisted = [
        {"role": "user", "content": "use MCP"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_mcp_1",
                "type": "function",
                "function": {"name": "mcp_docs_search", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_mcp_1", "content": "result"},
        {"role": "tool", "tool_call_id": "stale_mcp_call", "content": "stale"},
        {"role": "assistant", "content": "used MCP", "tool_calls": []},
        {"role": "user", "content": "follow up"},
    ]

    final_payload = sanitize_provider_request_kwargs({"messages": persisted})
    messages = final_payload["messages"]

    assert messages[1]["tool_calls"][0]["id"] == "call_mcp_1"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_mcp_1"
    assert not any(m.get("tool_call_id") == "stale_mcp_call" for m in messages)
    assert all(m.get("tool_calls") != [] for m in messages)
    assert persisted[3]["tool_call_id"] == "stale_mcp_call"


def test_interruptible_call_sanitizes_the_exact_kwargs_dispatched(monkeypatch):
    """Exercise the shared final call seam, including the direct/cron path."""
    from agent import chat_completion_helpers as helpers

    captured = {}

    def fake_direct(_agent, kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(helpers, "should_use_direct_api_call", lambda _agent: True)
    monkeypatch.setattr(helpers, "direct_api_call", fake_direct)
    persisted = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "done", "tool_calls": []},
        {"role": "user", "content": "next"},
    ]

    helpers.interruptible_api_call(object(), {"model": "deepseek", "messages": persisted})

    assert all(message.get("tool_calls") != [] for message in captured["messages"])
    assert persisted[1]["tool_calls"] == []


def test_reconstructed_responses_input_removes_orphan_output_without_mutation():
    reconstructed = [
        {"type": "message", "role": "user", "content": "go"},
        {"type": "function_call_output", "call_id": "orphan", "output": "old"},
        {"type": "function_call", "call_id": "valid", "name": "mcp_docs_search", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "valid", "output": "ok"},
    ]

    final_payload = sanitize_provider_request_kwargs({"input": reconstructed})

    assert [item.get("call_id") for item in final_payload["input"]] == [None, "valid", "valid"]
    assert reconstructed[1]["call_id"] == "orphan"

