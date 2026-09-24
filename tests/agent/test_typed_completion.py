"""The real Hermes loop completes through a caller-owned terminal tool.

Only the provider boundary and ordinary lookup handler are mocked. These tests
exercise native normalization, request construction, finalization and SQLite
history together, without provider traffic or a real profile.
"""

import copy
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.typed_completion import (
    TERMINAL_TOOL_NAME, TYPED_COMPLETION_GUIDANCE, TypedCompletionContract,
    TypedCompletionError, typed_prompt_cache_fingerprint,
)
from hermes_state import SessionDB
from run_agent import AIAgent


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"kind": {"enum": ["fact", "dialogue"]}, "record": {"type": "string"}},
    "required": ["kind", "record"],
}
VALUE = {"kind": "fact", "record": "record-1"}
LOOKUP = {"type": "function", "function": {
    "name": "lookup", "description": "Read a record",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}}


def call(name=TERMINAL_TOOL_NAME, arguments=None, call_id="complete-1"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(VALUE) if arguments is None else arguments,
    ))


def response(calls=None, text="", finish="tool_calls"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=text, tool_calls=calls), finish_reason=finish)],
        model="test/model", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    db = SessionDB(tmp_path / "sessions.db")
    agents = []

    def make(schema=SCHEMA, session_id="typed-session", mode="chat_completions", max_iterations=5,
             model="test/model", reasoning_config=None):
        with (patch("run_agent.get_tool_definitions", return_value=[copy.deepcopy(LOOKUP)]),
              patch("run_agent.check_toolset_requirements", return_value={}), patch("run_agent.OpenAI")):
            agent = AIAgent(api_key="test-key-only", base_url="https://example.invalid/v1", model=model,
                            quiet_mode=True, skip_context_files=True, skip_memory=True, tool_delay=0,
                            session_db=db, session_id=session_id, response_schema=schema,
                            max_iterations=max_iterations, reasoning_config=reasoning_config)
        agent.api_mode = mode
        agent.client = MagicMock()
        agent._cached_system_prompt = "Use lookup when needed, then complete the requested response."
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent._use_prompt_caching = False
        agents.append(agent)
        return agent

    yield make, db
    db.close()


def run(agent, replies, history=None):
    with (patch.object(agent, "_interruptible_api_call", side_effect=replies) as provider,
          patch("run_agent.handle_function_call", return_value='{"id":"record-1"}') as lookup,
          patch.object(agent, "_cleanup_task_resources"),
          patch.object(agent, "_handle_max_iterations") as summary):
        result = agent.run_conversation("Find my record", conversation_history=history)
    return result, provider, lookup, summary


@pytest.mark.parametrize("mode", ["chat_completions", "bedrock_converse"])
def test_lookup_then_terminal_is_one_loop_with_valid_history_and_stable_tools(make_agent, mode):
    make, db = make_agent
    agent = make(mode=mode)
    replies = [response([call("lookup", '{"query":"record"}', "lookup-1")], text="I will look it up."),
               response([call()])]
    result, provider, lookup, summary = run(agent, replies)
    assert result["typed_response"] == VALUE
    assert result["final_response"] == ""
    assert result["completed"] and not result["failed"]
    assert result["api_calls"] == 2 and provider.call_count == 2
    assert lookup.call_count == 1
    assert lookup.call_args.args[0] == "lookup"
    summary.assert_not_called()
    assert result["total_tokens"] == 30
    requests = [item.args[0] for item in provider.call_args_list]
    if mode == "bedrock_converse":
        assert all(item["toolConfig"]["toolChoice"] == {"any": {}} for item in requests)
        assert requests[0]["toolConfig"] == requests[1]["toolConfig"]
        assert requests[1]["messages"][-1]["content"][0]["toolResult"]["toolUseId"] == "lookup-1"
    else:
        assert all(item["tool_choice"] == "required" for item in requests)
        assert requests[0]["tools"] == requests[1]["tools"]
    assert [message["role"] for message in result["messages"]][-4:] == ["assistant", "tool", "assistant", "tool"]
    saved = db.get_messages_as_conversation(agent.session_id)
    assert saved[-2]["tool_calls"][0]["function"]["name"] == TERMINAL_TOOL_NAME
    assert saved[-1]["tool_call_id"] == saved[-2]["tool_calls"][0]["id"]
    assert json.loads(saved[-1]["content"]) == {"status": "completed"}


@pytest.mark.parametrize("reply", [
    response(None, text=json.dumps(VALUE), finish="stop"),
    response([call()], text="Here is your answer"),
    response([call(), call("lookup", '{"query":"record"}', "lookup-1")]),
    response([call(), call(call_id="complete-2")]),
    response([call(arguments='{"kind":"other","record":"record-1"}')]),
    response([call(arguments='{"kind":"fact","record":"x","prose":"invented"}')]),
    response([call(arguments='{"kind":"fact","kind":"dialogue","record":"x"}')]),
    response([call(arguments='{"kind":')]),
    response([call(arguments='{"kind":"fact","record":NaN}')]),
    response([call(arguments='{"kind":"fact","record":"x"}')], finish="length"),
    response([call(name="hermes_complete_respons")]),
    response([call(call_id="")]),
    response([call()], finish="content_filter"),
])
def test_invalid_or_mixed_terminal_result_fails_without_retry_or_tool_effect(make_agent, reply):
    make, _ = make_agent
    result, provider, lookup, summary = run(make(), [reply])
    assert result["failed"] and not result["completed"]
    assert result["turn_exit_reason"] == "typed_completion_invalid"
    assert result["typed_completion_error"].startswith("typed_completion_")
    assert "typed_response" not in result and result["final_response"] == ""
    assert provider.call_count == 1
    lookup.assert_not_called()
    summary.assert_not_called()


def test_budget_exhaustion_has_no_extra_prose_call(make_agent):
    make, _ = make_agent
    result, provider, lookup, summary = run(make(max_iterations=1), [response([call("lookup", '{"query":"x"}', "lookup-1")])])
    assert result["failed"] and "typed_response" not in result
    assert provider.call_count == 1 and lookup.call_count == 1
    summary.assert_not_called()


def test_malformed_provider_envelope_does_not_retry_or_fallback(make_agent):
    make, _ = make_agent
    agent = make()
    with patch.object(agent, "_try_activate_fallback") as fallback:
        result, provider, lookup, summary = run(agent, [SimpleNamespace(choices=[])])
    assert result["typed_completion_error"] == "typed_completion_invalid_response"
    assert result["failed"] and provider.call_count == 1
    lookup.assert_not_called()
    summary.assert_not_called()
    fallback.assert_not_called()


@pytest.mark.parametrize("mode, stop_reason", [
    ("chat_completions", None),
    ("chat_completions", "unknown_reason"),
    ("chat_completions", "length"),
    ("chat_completions", "content_filter"),
    ("anthropic_messages", None),
    ("anthropic_messages", "unknown_reason"),
    ("anthropic_messages", "max_tokens"),
    ("anthropic_messages", "model_context_window_exceeded"),
    ("anthropic_messages", "refusal"),
    ("anthropic_messages", "pause_turn"),
])
def test_unfinished_terminal_reason_never_completes_retries_or_executes_tools(make_agent, mode, stop_reason):
    make, db = make_agent
    agent = make(mode=mode)
    raw = response([call()], finish=stop_reason)
    if mode == "anthropic_messages":
        raw = SimpleNamespace(
            role="assistant", stop_reason=stop_reason,
            content=[SimpleNamespace(type="tool_use", id="complete-1", name=TERMINAL_TOOL_NAME, input=VALUE)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
    with patch.object(agent, "_try_activate_fallback") as fallback:
        result, provider, lookup, summary = run(agent, [raw])
    assert result["failed"] and not result["completed"]
    assert "typed_response" not in result and result["final_response"] == ""
    assert result["typed_completion_error"].startswith("typed_completion_")
    assert provider.call_count == 1
    assert all(message["role"] != "tool" for message in db.get_messages_as_conversation(agent.session_id))
    lookup.assert_not_called()
    summary.assert_not_called()
    fallback.assert_not_called()


@pytest.mark.parametrize("mode", ["chat_completions", "anthropic_messages"])
def test_missing_stop_reason_attribute_is_rejected_by_strict_normalization(mode):
    from agent.transports.anthropic import AnthropicTransport
    from agent.transports.chat_completions import ChatCompletionsTransport

    if mode == "chat_completions":
        raw = response([call()])
        del raw.choices[0].finish_reason
        transport = ChatCompletionsTransport()
    else:
        raw = SimpleNamespace(role="assistant", content=[
            SimpleNamespace(type="tool_use", id="complete-1", name=TERMINAL_TOOL_NAME, input=VALUE),
        ])
        transport = AnthropicTransport()
    with pytest.raises(TypedCompletionError, match="invalid_envelope"):
        transport.normalize_response(raw, strict_tools=True)


@pytest.mark.parametrize("mode, stop_reason", [
    ("chat_completions", None),
    ("chat_completions", "unknown_reason"),
    ("anthropic_messages", None),
    ("anthropic_messages", "unknown_reason"),
])
def test_untyped_stop_reason_normalization_preserves_existing_behavior(mode, stop_reason):
    from agent.transports.anthropic import AnthropicTransport
    from agent.transports.chat_completions import ChatCompletionsTransport

    if mode == "chat_completions":
        raw = response(None, text="Hello", finish=stop_reason)
        normalized = ChatCompletionsTransport().normalize_response(raw)
        assert normalized.finish_reason == (stop_reason or "stop")
    else:
        raw = SimpleNamespace(role="assistant", content=[SimpleNamespace(type="text", text="Hello")],
                              stop_reason=stop_reason)
        normalized = AnthropicTransport().normalize_response(raw)
        assert normalized.finish_reason == "stop"
    assert normalized.content == "Hello"


@pytest.mark.parametrize("stop_reason, arguments, completed", [
    (None, json.dumps(VALUE), False),
    ("unknown_reason", json.dumps(VALUE), False),
    ("length", json.dumps(VALUE), False),
    ("tool_calls", '{"kind":"fact","record":"record-1",}', False),
    ("tool_calls", json.dumps(VALUE), True),
])
def test_actual_chat_stream_preserves_typed_arguments_and_requires_terminal_reason(
    make_agent, stop_reason, arguments, completed,
):
    make, db = make_agent
    agent = make()
    agent.client = None  # Exercise the actual collector, not the mock-client non-streaming path.
    client = MagicMock()
    client.chat.completions.create.return_value = iter([
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(role="assistant", content=None, tool_calls=[SimpleNamespace(
                index=0, id="complete-1", type="function",
                function=SimpleNamespace(name=TERMINAL_TOOL_NAME, arguments=arguments),
            )]), finish_reason=stop_reason,
        )]),
    ])
    with (patch.object(agent, "_create_request_openai_client", return_value=client),
          patch("agent.chat_completion_helpers._repair_tool_call_arguments") as repair,
          patch("run_agent.handle_function_call") as lookup,
          patch.object(agent, "_cleanup_task_resources"),
          patch.object(agent, "_try_activate_fallback") as fallback,
          patch.object(agent, "_handle_max_iterations") as summary):
        result = agent.run_conversation("Select the saved record")
    assert result["completed"] is completed and result["failed"] is not completed
    assert result["final_response"] == ""
    if completed:
        assert result["typed_response"] == VALUE
        assert db.get_messages_as_conversation(agent.session_id)[-1]["tool_call_id"] == "complete-1"
    else:
        assert "typed_response" not in result
        assert result["typed_completion_error"].startswith("typed_completion_")
        assert all(message["role"] != "tool" for message in db.get_messages_as_conversation(agent.session_id))
    assert client.chat.completions.create.call_count == 1
    repair.assert_not_called()
    lookup.assert_not_called()
    fallback.assert_not_called()
    summary.assert_not_called()


def test_invalid_completion_after_lookup_does_not_request_format_repair(make_agent):
    make, _ = make_agent
    result, provider, lookup, _ = run(make(), [response([call("lookup", '{"query":"record"}', "lookup-1")]),
                                             response(None, text='<response>{"kind":"fact","record":"record-1"}</response>', finish="stop")])
    assert result["failed"] and "typed_response" not in result
    assert provider.call_count == 2 and lookup.call_count == 1


def test_terminal_at_last_allowed_iteration_is_success(make_agent):
    make, _ = make_agent
    result, provider, _, _ = run(make(max_iterations=1), [response([call()])])
    assert result["completed"] and not result["failed"] and provider.call_count == 1


def test_schema_is_stable_across_native_turns_and_history_replay(make_agent):
    make, db = make_agent
    first, _, _, _ = run(make(), [response([call()])])
    saved = db.get_messages_as_conversation("typed-session")
    second, provider, _, _ = run(make(), [response([call(call_id="complete-2")])], saved)
    assert second["typed_response"] == first["typed_response"]
    assert provider.call_args.args[0]["messages"][-2]["tool_call_id"] == "complete-1"
    before = db.get_messages("typed-session")
    changed = copy.deepcopy(SCHEMA)
    changed["properties"]["kind"]["enum"] = ["anything"]
    for schema in (changed, None):
        agent = make(schema=schema)
        with patch.object(agent, "_interruptible_api_call") as never:
            with pytest.raises(ValueError, match="response_schema_changed"):
                agent.run_conversation("Change my rules", conversation_history=saved)
            never.assert_not_called()
    assert db.get_messages("typed-session") == before


@pytest.mark.parametrize("typed", [False, True])
def test_actual_api_prompt_uses_contract_presentation_and_preserves_identity_and_safety(make_agent, typed):
    from agent.prompt_builder import PLATFORM_HINTS, TASK_COMPLETION_GUIDANCE

    make, db = make_agent
    agent = make(schema=SCHEMA if typed else None)
    agent.platform = "api_server"
    agent.load_soul_identity = True
    agent._cached_system_prompt = None  # Use the real prompt builder.
    agent.ephemeral_system_prompt = "Caller task: select the relevant saved record."
    replies = ([response([call("lookup", '{"query":"record"}', "lookup-1")]), response([call()])]
               if typed else [response(None, "Hello", "stop")])
    with patch("run_agent.load_soul_md", return_value="Synthetic application identity."):
        result, provider, lookup, _ = run(agent, replies)
    assert result["completed"]
    assert provider.call_count == (2 if typed else 1)
    assert lookup.call_count == (1 if typed else 0)
    prompts = [request.args[0]["messages"][0]["content"] for request in provider.call_args_list]
    assert all(prompt == prompts[0] for prompt in prompts)
    stored = db.get_session(agent.session_id)
    assert agent.ephemeral_system_prompt not in stored["system_prompt"]
    for sent in prompts:
        assert "Synthetic application identity." in sent
        assert TASK_COMPLETION_GUIDANCE in sent
        assert "Do not modify another profile" in sent
        assert agent.ephemeral_system_prompt in sent
        assert sent.count(TYPED_COMPLETION_GUIDANCE) == (1 if typed else 0)
        assert (PLATFORM_HINTS["api_server"] in sent) is not typed
    if typed:
        assert result["typed_response"] == VALUE
        assert stored["system_prompt_contract"] == typed_prompt_cache_fingerprint(SCHEMA, stored["system_prompt"])
        assert agent.valid_tool_names == {"lookup", TERMINAL_TOOL_NAME}
    else:
        assert stored["system_prompt_contract"] is None
        assert agent.valid_tool_names == {"lookup"}


def test_legacy_adoption_rebuilds_actual_prompt_once_and_restores_after_restart(make_agent, tmp_path):
    from agent.prompt_builder import PLATFORM_HINTS

    make, db = make_agent
    sid = "legacy-prompt"
    old_prompt = "Old identity.\n\n" + PLATFORM_HINTS["api_server"]
    db.create_session(sid, "api_server", system_prompt=old_prompt)
    db.append_message(sid, "user", "Earlier request")
    db.append_message(sid, "assistant", "Earlier answer")
    before = db.get_messages(sid)
    history = db.get_messages_as_conversation(sid)
    first = make(session_id=sid)
    first.platform = "api_server"
    first._cached_system_prompt = old_prompt
    result, provider, _, _ = run(first, [response([call()])], history)
    assert result["completed"]
    saved = db.get_session(sid)
    assert TYPED_COMPLETION_GUIDANCE in saved["system_prompt"]
    assert PLATFORM_HINTS["api_server"] not in saved["system_prompt"]
    assert saved["system_prompt"] != old_prompt
    assert saved["system_prompt_contract"] == typed_prompt_cache_fingerprint(SCHEMA, saved["system_prompt"])
    assert db.get_messages(sid)[:len(before)] == before
    first_prefix = provider.call_args.args[0]["messages"][0]["content"]
    history = db.get_messages_as_conversation(sid)
    second = make(session_id=sid)
    second.platform = "api_server"
    second._cached_system_prompt = None
    db.close()
    reopened = SessionDB(tmp_path / "sessions.db")
    try:
        second._session_db = reopened
        with patch.object(second, "_build_system_prompt", side_effect=AssertionError("must restore the exact prefix")):
            result, provider, _, _ = run(second, [response([call(call_id="complete-2")])], history)
        assert result["completed"]
        assert provider.call_args.args[0]["messages"][0]["content"] == first_prefix
        assert reopened.get_session(sid)["system_prompt_contract"] == saved["system_prompt_contract"]
    finally:
        reopened.close()


def test_unversioned_typed_cache_and_old_writer_prompt_changes_invalidate_only_cache(tmp_path):
    path = tmp_path / "legacy-typed-prompt.db"
    db = SessionDB(path)
    sid = "legacy-typed"
    db.create_session(sid, "api_server", system_prompt="Legacy natural-text prompt")
    db.append_message(sid, "user", "Saved request")
    before = db.get_messages(sid)
    db.close()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE sessions SET response_schema = ? WHERE id = ?", (json.dumps(SCHEMA, sort_keys=True, separators=(",", ":")), sid))
        connection.execute("ALTER TABLE sessions DROP COLUMN system_prompt_contract")
    db = SessionDB(path)
    try:
        assert db.get_messages(sid) == before
        assert db.bind_response_schema(sid, SCHEMA) is True
        assert db.get_session(sid)["system_prompt"] is None
        assert db.get_messages(sid) == before
        prompt = "Identity and task rules.\n\n" + TYPED_COMPLETION_GUIDANCE
        db.update_system_prompt(sid, prompt)
        assert db.bind_response_schema(sid, SCHEMA) is False
        # An older writer uses the unchanged session/message columns. It does
        # not know the new nullable provenance column or update it.
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE sessions SET system_prompt = ? WHERE id = ?", ("Prompt written by an older native runtime", sid))
            assert connection.execute("SELECT id, source, response_schema FROM sessions WHERE id = ?", (sid,)).fetchone()[:2] == (sid, "api_server")
            assert connection.execute("SELECT role, content FROM messages WHERE session_id = ?", (sid,)).fetchall() == [("user", "Saved request")]
        assert db.bind_response_schema(sid, SCHEMA) is True
        assert db.get_session(sid)["system_prompt"] is None
        assert db.get_messages(sid) == before
    finally:
        db.close()


def test_prompt_contract_revision_rebuilds_without_changing_schema_or_messages(make_agent):
    make, db = make_agent
    agent = make()
    agent.platform = "api_server"
    result, _, _, _ = run(agent, [response([call()])])
    assert result["completed"]
    saved = db.get_session(agent.session_id)
    before = db.get_messages(agent.session_id)
    with patch("agent.typed_completion.TYPED_COMPLETION_PROMPT_VERSION", "hermes.typed-completion-prompt.next"):
        assert db.bind_response_schema(agent.session_id, SCHEMA) is True
        assert db.get_session(agent.session_id)["system_prompt"] is None
    assert db.get_session(agent.session_id)["response_schema"] == saved["response_schema"]
    assert db.get_messages(agent.session_id) == before


@pytest.mark.parametrize("cache_is_binary", [False, True])
def test_restore_checks_exact_prompt_provenance_before_accepting_cached_text(make_agent, tmp_path, cache_is_binary):
    from agent.conversation_loop import _restore_or_build_system_prompt
    from agent.prompt_builder import PLATFORM_HINTS

    make, db = make_agent
    agent = make()
    agent.platform = "api_server"
    run(agent, [response([call()])])
    before = db.get_messages(agent.session_id)
    with sqlite3.connect(tmp_path / "sessions.db") as connection:
        connection.execute("UPDATE sessions SET system_prompt = ? WHERE id = ?",
                           (b"Legacy cache bytes" if cache_is_binary else PLATFORM_HINTS["api_server"], agent.session_id))
    agent._cached_system_prompt = None
    # Exercise restore itself, independently of the earlier schema-bind check.
    _restore_or_build_system_prompt(agent, None, db.get_messages_as_conversation(agent.session_id))
    assert TYPED_COMPLETION_GUIDANCE in agent._cached_system_prompt
    assert PLATFORM_HINTS["api_server"] not in agent._cached_system_prompt
    saved = db.get_session(agent.session_id)
    assert saved["system_prompt_contract"] == typed_prompt_cache_fingerprint(SCHEMA, saved["system_prompt"])
    assert db.get_messages(agent.session_id) == before


def test_actual_compression_rebuilds_and_stamps_the_inherited_typed_contract(make_agent):
    from agent.prompt_builder import PLATFORM_HINTS

    make, db = make_agent
    agent = make()
    agent.platform = "api_server"
    result, _, _, _ = run(agent, [response([call()])])
    old_sid = agent.session_id
    old_messages = db.get_messages(old_sid)
    agent._compression_feasibility_checked = True
    compressed = [{"role": "user", "content": "Retained task context."}]
    with (patch.object(agent.context_compressor, "compress", return_value=compressed),
          patch.object(agent, "commit_memory_session")):
        messages, prompt = agent._compress_context(result["messages"], None, approx_tokens=1000, force=True)
    assert messages == compressed and agent.session_id != old_sid
    child = db.get_session(agent.session_id)
    assert child["parent_session_id"] == old_sid
    assert json.loads(child["response_schema"]) == SCHEMA
    assert child["system_prompt"] == prompt
    assert TYPED_COMPLETION_GUIDANCE in prompt and PLATFORM_HINTS["api_server"] not in prompt
    assert child["system_prompt_contract"] == typed_prompt_cache_fingerprint(SCHEMA, prompt)
    assert db.bind_response_schema(agent.session_id, SCHEMA) is False
    assert db.get_messages(old_sid) == old_messages


def test_freeform_adoption_preserves_the_exact_cached_prompt(make_agent):
    _, db = make_agent
    sid = "legacy-freeform"
    db.create_session(sid, "api_server", system_prompt="Exact original freeform prompt")
    db.append_message(sid, "user", "Prior message")
    before = db.get_messages(sid)
    assert db.bind_response_schema(sid, None) is False
    assert db.get_session(sid)["system_prompt"] == "Exact original freeform prompt"
    assert db.get_session(sid)["system_prompt_contract"] is None
    assert db.get_messages(sid) == before


def test_legacy_contract_adoption_preserves_every_message_and_compression_inherits(make_agent):
    _, db = make_agent
    db.create_session("legacy", "api_server")
    db.append_message("legacy", "user", "Earlier request")
    db.append_message("legacy", "assistant", "Earlier freeform answer")
    before = db.get_messages("legacy")
    db.bind_response_schema("legacy", SCHEMA)
    assert db.get_messages("legacy") == before
    db.create_session("delegated", "api_server", parent_session_id="legacy")
    assert db.get_session("delegated")["response_schema"] is None
    db.end_session("legacy", "compression")
    db.create_session("compressed", "api_server", parent_session_id="legacy")
    db.bind_response_schema("compressed", SCHEMA)
    with pytest.raises(ValueError, match="response_schema_changed"):
        db.bind_response_schema("compressed", None)


@pytest.mark.parametrize("schema", [None, [], {"type":"string"},
    {"type":"object", "$ref":"https://example.invalid/schema"},
    {"type":"object", "properties":{"x":{"$ref":"#/$defs/x"}}},
    {"type":"object", "required":"wrong"},
    {"type":"object", "description":"x" * 65536},
])
def test_invalid_or_unbounded_caller_schema_is_rejected(schema):
    with pytest.raises(TypedCompletionError):
        TypedCompletionContract(schema)


def test_schema_definition_is_copied(make_agent):
    make, _ = make_agent
    schema = copy.deepcopy(SCHEMA)
    agent = make(schema=schema)
    schema["properties"]["kind"]["enum"].append("invented")
    assert agent._typed_completion_contract.schema == SCHEMA
    assert agent.tools[-1]["function"]["parameters"] == SCHEMA


def test_unsupported_transport_fails_before_native_turn(make_agent):
    make, _ = make_agent
    agent = make(mode="codex_app_server")
    with patch.object(agent, "_run_codex_app_server_turn") as alternate:
        with pytest.raises(TypedCompletionError, match="transport_unsupported"):
            agent.run_conversation("hello")
        alternate.assert_not_called()


def test_reserved_tool_collision_fails_at_construction(monkeypatch):
    existing = copy.deepcopy(LOOKUP)
    existing["function"]["name"] = TERMINAL_TOOL_NAME
    with (patch("run_agent.get_tool_definitions", return_value=[existing]),
          patch("run_agent.check_toolset_requirements", return_value={}), patch("run_agent.OpenAI")):
        with pytest.raises(TypedCompletionError, match="tool_collision"):
            AIAgent(api_key="test-key-only", base_url="https://example.invalid/v1", model="test/model",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, response_schema=SCHEMA)


def test_self_contained_one_of_contract_reaches_bedrock_unchanged(make_agent):
    make, _ = make_agent
    schema = {"type": "object", "oneOf": [
        {"type": "object", "additionalProperties": False, "properties": {"kind": {"const": "reply"}, "act": {"enum": ["help", "acknowledge"]}}, "required": ["kind", "act"]},
        {"type": "object", "additionalProperties": False, "properties": {"kind": {"const": "answer"}, "records": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "string"}}}, "required": ["kind", "records"]},
        {"type": "object", "additionalProperties": False, "properties": {"kind": {"const": "unknown"}}, "required": ["kind"]},
    ]}
    agent = make(schema=schema, mode="bedrock_converse")
    value = {"kind": "reply", "act": "help"}
    result, provider, _, _ = run(agent, [response([call(arguments=json.dumps(value))])])
    assert result["typed_response"] == value
    tool = provider.call_args.args[0]["toolConfig"]["tools"][-1]["toolSpec"]
    assert tool["inputSchema"]["json"] == schema
    for invalid in ({"kind":"reply"}, {"kind":"reply", "act":"invented"}, {"kind":"unknown", "text":"prose"}, {"kind":"answer", "records":[]}):
        with pytest.raises(TypedCompletionError, match="schema_mismatch"):
            agent._typed_completion_contract.validate_value(invalid)


def bedrock_reply(blocks, stop="tool_use"):
    return {"output": {"message": {"role": "assistant", "content": blocks}}, "stopReason": stop,
            "usage": {"inputTokens": 10, "outputTokens": 5}}


def bedrock_tool(name=TERMINAL_TOOL_NAME, value=VALUE, call_id="terminal-1"):
    return {"toolUse": {"toolUseId": call_id, "name": name, "input": value}}


def bedrock_events(arguments, name=TERMINAL_TOOL_NAME, call_id="terminal-1"):
    return {"stream": iter([
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": call_id, "name": name}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": arguments}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
    ])}


def run_bedrock(agent, replies, streaming=False):
    client = MagicMock()
    if streaming:
        agent.client = None  # Exercise the actual internal ConverseStream collector.
        client.converse_stream.side_effect = replies
    else:
        client.converse.side_effect = replies
    with (patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client),
          patch("run_agent.handle_function_call", return_value='{"id":"record-1"}') as lookup,
          patch.object(agent, "_cleanup_task_resources"),
          patch.object(agent, "_try_activate_fallback") as fallback,
          patch.object(agent, "_handle_max_iterations") as summary):
        result = agent.run_conversation("Find a record")
    fallback.assert_not_called()
    summary.assert_not_called()
    return result, client, lookup


@pytest.mark.parametrize("streaming", [False, True])
def test_real_bedrock_envelope_after_lookup_completes_and_preserves_usage(make_agent, streaming):
    make, db = make_agent
    agent = make(mode="bedrock_converse")
    if streaming:
        replies = [bedrock_events('{"query":"record"}', "lookup", "lookup-1"), bedrock_events(json.dumps(VALUE))]
    else:
        replies = [bedrock_reply([bedrock_tool("lookup", {"query":"record"}, "lookup-1")]), bedrock_reply([bedrock_tool()])]
    result, client, lookup = run_bedrock(agent, replies, streaming)
    assert result["typed_response"] == VALUE and result["completed"]
    assert result["total_tokens"] == 30 and result["api_calls"] == 2
    assert lookup.call_count == 1
    provider = client.converse_stream if streaming else client.converse
    assert provider.call_count == 2
    assert provider.call_args.kwargs["toolConfig"]["toolChoice"] == {"any": {}}
    assert db.get_messages_as_conversation(agent.session_id)[-1]["tool_call_id"] == "terminal-1"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "none"])
def test_configured_nova2_reasoning_survives_lookup_and_typed_completion(make_agent, tmp_path, monkeypatch, streaming, effort):
    import gateway.run as gateway_run

    config_home = tmp_path / "home"
    config_home.mkdir(exist_ok=True)
    (config_home / "config.yaml").write_text(f"agent:\n  reasoning_effort: {effort}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", config_home)
    config = gateway_run.GatewayRunner._load_reasoning_config()
    make, db = make_agent
    agent = make(mode="bedrock_converse", model="us.amazon.nova-2-lite-v1:0", reasoning_config=config)
    agent.max_tokens = 8192
    if streaming:
        replies = [bedrock_events('{"query":"record"}', "lookup", "lookup-1"), bedrock_events(json.dumps(VALUE))]
    else:
        replies = [bedrock_reply([bedrock_tool("lookup", {"query":"record"}, "lookup-1")]), bedrock_reply([bedrock_tool()])]
    result, client, lookup = run_bedrock(agent, replies, streaming)
    provider = client.converse_stream if streaming else client.converse
    expected = {"type": "disabled"} if effort == "none" else {"type": "enabled", "maxReasoningEffort": effort}
    assert result["completed"] and result["typed_response"] == VALUE
    assert provider.call_count == 2 and lookup.call_count == 1
    requests = [call.kwargs for call in provider.call_args_list]
    for request in requests:
        assert request["additionalModelRequestFields"] == {"reasoningConfig": expected}
        assert request["toolConfig"]["toolChoice"] == {"any": {}}
        assert {tool["toolSpec"]["name"] for tool in request["toolConfig"]["tools"]} == {"lookup", TERMINAL_TOOL_NAME}
        if effort == "high":
            assert "inferenceConfig" not in request
        else:
            assert request["inferenceConfig"] == {"maxTokens": 8192}
    assert requests[0]["system"] == requests[1]["system"]
    assert requests[0]["toolConfig"] == requests[1]["toolConfig"]
    assert result["api_calls"] == 2 and result["total_tokens"] == 30
    assert db.get_messages_as_conversation(agent.session_id)[-1]["tool_call_id"] == "terminal-1"


@pytest.mark.parametrize("blocks", [
    [bedrock_tool(), bedrock_tool("lookup", {"query":"record"}, "lookup-1")],
    [bedrock_tool(), {"text":"invented answer"}],
    [bedrock_tool(), {"text":"", "toolUse":{"toolUseId":"hidden", "name":"lookup", "input":{}}}],
    [{"toolUse":{"toolUseId":"terminal-1", "name":TERMINAL_TOOL_NAME}}],
    [bedrock_tool(value=None)],
    [bedrock_tool(value='{"kind":"fact","record":"record-1"}')],
    [bedrock_tool(), {"unknownBlock":{}}],
])
def test_raw_bedrock_malformed_or_mixed_terminal_never_executes_tools(make_agent, blocks):
    make, _ = make_agent
    result, client, lookup = run_bedrock(make(mode="bedrock_converse"), [bedrock_reply(blocks)])
    assert result["failed"] and "typed_response" not in result and result["final_response"] == ""
    assert client.converse.call_count == 1
    lookup.assert_not_called()


@pytest.mark.parametrize("arguments", ["", "{", '{"a":1,"a":2}', "null", "NaN"])
def test_bedrock_stream_never_repairs_terminal_arguments_to_empty_object(make_agent, arguments):
    make, _ = make_agent
    agent = make(schema={"type":"object"}, mode="bedrock_converse")
    result, client, lookup = run_bedrock(agent, [bedrock_events(arguments)], streaming=True)
    assert result["failed"] and "typed_response" not in result
    assert client.converse_stream.call_count == 1
    lookup.assert_not_called()


def test_bedrock_stream_without_terminal_stop_fails_once(make_agent):
    make, _ = make_agent
    events = list(bedrock_events(json.dumps(VALUE))["stream"])
    events = [event for event in events if "messageStop" not in event]
    result, client, lookup = run_bedrock(make(mode="bedrock_converse"), [{"stream": iter(events)}], streaming=True)
    assert result["typed_completion_error"] == "typed_completion_invalid_envelope"
    assert client.converse_stream.call_count == 1
    lookup.assert_not_called()


@pytest.mark.parametrize("extra", ["text", "tool", "unknown"])
def test_anthropic_normalization_cannot_hide_mixed_terminal_blocks(extra):
    from agent.transports.anthropic import AnthropicTransport
    blocks = [SimpleNamespace(type="tool_use", id="terminal-1", name=TERMINAL_TOOL_NAME, input=VALUE)]
    if extra == "text":
        blocks.append(SimpleNamespace(type="text", text="invented"))
    elif extra == "tool":
        blocks.append(SimpleNamespace(type="tool_use", id="lookup-1", name="lookup", input={"query":"record"}))
    else:
        blocks.append(SimpleNamespace(type="server_tool_use", id="hidden", name="effect", input={}))
    raw = SimpleNamespace(role="assistant", content=blocks, stop_reason="tool_use")
    with pytest.raises(TypedCompletionError):
        normalized = AnthropicTransport().normalize_response(raw, strict_tools=True)
        TypedCompletionContract(SCHEMA).inspect_response(normalized, {TERMINAL_TOOL_NAME, "lookup"})


def test_anthropic_terminal_uses_same_validation_and_no_prose_conversion():
    from agent.transports.anthropic import AnthropicTransport
    raw = SimpleNamespace(role="assistant", content=[SimpleNamespace(type="tool_use", id="terminal-1", name=TERMINAL_TOOL_NAME, input=VALUE)], stop_reason="tool_use")
    normalized = AnthropicTransport().normalize_response(raw, strict_tools=True)
    assert TypedCompletionContract(SCHEMA).inspect_response(normalized, {TERMINAL_TOOL_NAME}) == VALUE


def test_binding_survives_restart_and_old_schema_upgrade(tmp_path):
    path = tmp_path / "legacy.db"
    db = SessionDB(path)
    db.create_session("legacy", "api_server")
    db.append_message("legacy", "user", "Old history")
    old_rows = db.get_messages("legacy")
    db.close()
    # Simulate the deployed pre-feature schema without changing its data.
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE sessions DROP COLUMN response_schema")
    db = SessionDB(path)
    db.bind_response_schema("legacy", SCHEMA)
    assert db.get_messages("legacy") == old_rows
    db.close()
    db = SessionDB(path)
    try:
        db.bind_response_schema("legacy", copy.deepcopy(SCHEMA))
        with pytest.raises(ValueError, match="response_schema_changed"):
            db.bind_response_schema("legacy", {"type": "object"})
        assert db.get_messages("legacy") == old_rows
    finally:
        db.close()


def test_freeform_response_is_unchanged(make_agent):
    make, _ = make_agent
    result, provider, _, _ = run(make(schema=None), [response(None, text="Hello", finish="stop")])
    assert result["final_response"] == "Hello" and "typed_response" not in result
    assert "tool_choice" not in provider.call_args.args[0]
    assert [tool["function"]["name"] for tool in provider.call_args.args[0]["tools"]] == ["lookup"]
