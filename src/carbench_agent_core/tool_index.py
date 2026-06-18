"""Utilities for validating CAR-bench tool availability and arguments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .response_renderer import (
    render_malformed_tool_arguments,
    render_missing_capability,
    render_missing_required_information,
    render_unavailable_control,
)


@dataclass(frozen=True)
class ToolIndex:
    """Small read-only index over the currently available tool schema."""

    tools: list[dict[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_by_name",
            {
                tool.get("function", {}).get("name"): tool
                for tool in self.tools
                if tool.get("function", {}).get("name")
            },
        )

    @property
    def names(self) -> set[str]:
        return set(self._by_name)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> dict[str, Any] | None:
        return self._by_name.get(name)

    def required_args(self, name: str) -> set[str]:
        tool = self.get(name)
        if not tool:
            return set()
        params = tool.get("function", {}).get("parameters", {})
        required = params.get("required") or []
        return {arg for arg in required if isinstance(arg, str)}

    def arg_names(self, name: str) -> set[str]:
        tool = self.get(name)
        if not tool:
            return set()
        params = tool.get("function", {}).get("parameters", {})
        properties = params.get("properties") or {}
        return {arg for arg in properties if isinstance(arg, str)}

    def arg_schema(self, name: str, arg: str) -> dict[str, Any]:
        tool = self.get(name)
        if not tool:
            return {}
        params = tool.get("function", {}).get("parameters", {})
        properties = params.get("properties") or {}
        schema = properties.get(arg)
        return schema if isinstance(schema, dict) else {}

    def validate_call(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Return a user-safe validation error or None if the call is available."""

        if not self.has(name):
            return render_missing_capability()

        if not isinstance(arguments, dict):
            return render_malformed_tool_arguments()

        required_missing = sorted(self.required_args(name) - set(arguments))
        if required_missing:
            return render_missing_required_information()

        allowed = self.arg_names(name)
        tool = self.get(name) or {}
        params = tool.get("function", {}).get("parameters", {})
        properties = params.get("properties")
        if isinstance(properties, dict):
            unknown = sorted(set(arguments) - allowed)
            if unknown and (
                allowed
                or "" in unknown
                or params.get("additionalProperties") is False
            ):
                return render_unavailable_control()

        for arg_name, value in arguments.items():
            schema = self.arg_schema(name, arg_name)
            if not schema:
                continue
            enum_values = schema.get("enum")
            if isinstance(enum_values, list) and value not in enum_values:
                return render_unavailable_control()

            schema_type = schema.get("type")
            if isinstance(schema_type, str) and not _matches_schema_type(
                value, schema_type
            ):
                return render_unavailable_control()

        return None


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any] | None:
    """Parse provider-specific tool-call arguments into a dictionary."""

    if isinstance(raw_arguments, dict):
        return raw_arguments
    if raw_arguments is None:
        return {}
    if not isinstance(raw_arguments, str):
        return None
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _matches_schema_type(value: Any, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "null":
        return value is None
    return True
