"""Caller-owned typed completion for the normal Hermes tool loop.

The model selects data; it cannot define or relax this contract. This is a
terminal protocol operation, never a tool-registry handler or a second model
request. Schemas are bounded, self-contained JSON Schema 2020-12 objects.
"""

from __future__ import annotations

import json
from typing import Any

TERMINAL_TOOL_NAME = "hermes_complete_response"
SUPPORTED_TRANSPORTS = frozenset({"bedrock_converse", "chat_completions", "anthropic_messages"})
MAX_JSON_BYTES = 65_536
MAX_JSON_NODES = 4_096
MAX_JSON_DEPTH = 32


class TypedCompletionError(ValueError):
    """Invalid caller contract or model completion; safe, data-free code."""


def canonical_json(value: Any) -> str:
    """Bound and copy JSON without NaN, coercion, or unbounded nesting."""
    remaining = MAX_JSON_NODES

    def visit(node, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_JSON_DEPTH:
            raise TypedCompletionError("typed_completion_json_limit")
        if type(node) is dict:
            if any(type(key) is not str for key in node):
                raise TypedCompletionError("typed_completion_invalid_json")
            for child in node.values():
                visit(child, depth + 1)
        elif type(node) is list:
            for child in node:
                visit(child, depth + 1)
        elif node is not None and type(node) not in (str, int, float, bool):
            raise TypedCompletionError("typed_completion_invalid_json")

    visit(value)
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise TypedCompletionError("typed_completion_invalid_json") from exc
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise TypedCompletionError("typed_completion_json_limit")
    return encoded


class TypedCompletionContract:
    def __init__(self, schema: dict):
        # Optional dependency, but mandatory when this feature is selected.
        # Never silently accept unvalidated output when it is not installed.
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:
            raise TypedCompletionError("typed_completion_dependency_missing") from exc
        if type(schema) is not dict or schema.get("type") != "object":
            raise TypedCompletionError("response_schema_must_be_object")
        self.schema_json = canonical_json(schema)
        self._schema = json.loads(self.schema_json)

        def self_contained(node):
            if isinstance(node, dict):
                if any(key in node for key in ("$ref", "$dynamicRef", "$recursiveRef", "$id")):
                    raise TypedCompletionError("response_schema_references_unsupported")
                if "$schema" in node and node["$schema"] != "https://json-schema.org/draft/2020-12/schema":
                    raise TypedCompletionError("response_schema_draft_unsupported")
                for child in node.values():
                    self_contained(child)
            elif isinstance(node, list):
                for child in node:
                    self_contained(child)

        self_contained(self._schema)
        try:
            Draft202012Validator.check_schema(self._schema)
        except Exception as exc:
            raise TypedCompletionError("response_schema_invalid") from exc
        self._validator = Draft202012Validator(self._schema)

    @property
    def schema(self):
        return json.loads(self.schema_json)

    def tool_definition(self):
        return {"type": "function", "function": {
            "name": TERMINAL_TOOL_NAME,
            "description": (
                "Complete this turn with the caller's typed response. Use the other tools "
                "first when information is needed. Call this tool alone, with no other "
                "tool calls or assistant text. Its arguments are the final response."
            ),
            "parameters": self.schema,
        }}

    def validate_value(self, value):
        value = json.loads(canonical_json(value))
        if type(value) is not dict or not self._validator.is_valid(value):
            raise TypedCompletionError("typed_completion_schema_mismatch")
        return value

    def inspect_response(self, response, valid_tool_names):
        """Return the terminal value, or None for a valid ordinary tool batch.

        Malformed calls, final prose, truncation, and mixed terminal batches fail
        before any tool executes. No fuzzy tool-name or JSON repair is used.
        """
        calls = response.tool_calls or []
        if response.finish_reason not in {"stop", "tool_calls"} or not calls:
            raise TypedCompletionError("typed_completion_missing")
        ids = [call.id for call in calls]
        if any(not isinstance(call_id, str) or not call_id for call_id in ids) or len(set(ids)) != len(ids):
            raise TypedCompletionError("typed_completion_invalid_call_id")
        terminal = [call for call in calls if call.function.name == TERMINAL_TOOL_NAME]
        if terminal and len(calls) != 1:
            raise TypedCompletionError("typed_completion_mixed_tools")
        if terminal and response.content not in (None, ""):
            raise TypedCompletionError("typed_completion_mixed_content")

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise TypedCompletionError("typed_completion_duplicate_key")
                result[key] = value
            return result

        decoded = []
        for call in calls:
            if call.function.name not in valid_tool_names:
                raise TypedCompletionError("typed_completion_unknown_tool")
            arguments = call.function.arguments
            if not isinstance(arguments, str) or len(arguments.encode("utf-8")) > MAX_JSON_BYTES:
                raise TypedCompletionError("typed_completion_invalid_arguments")
            try:
                value = json.loads(arguments, object_pairs_hook=unique_object)
                canonical_json(value)
            except (ValueError, TypeError, RecursionError) as exc:
                raise TypedCompletionError("typed_completion_invalid_arguments") from exc
            if type(value) is not dict:
                raise TypedCompletionError("typed_completion_invalid_arguments")
            decoded.append(value)
        return self.validate_value(decoded[0]) if terminal else None
