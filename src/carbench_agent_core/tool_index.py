"""Utilities for validating CAR-bench tool availability and arguments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


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

    def validate_call(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Return a user-safe validation error or None if the call is available."""

        if not self.has(name):
            return f"I can't do that because the {name} capability is not available right now."

        if not isinstance(arguments, dict):
            return f"I can't use {name} because its arguments are malformed."

        required_missing = sorted(self.required_args(name) - set(arguments))
        if required_missing:
            missing = ", ".join(required_missing)
            return f"I can't use {name} yet because the required {missing} information is missing."

        allowed = self.arg_names(name)
        if allowed:
            unknown = sorted(set(arguments) - allowed)
            if unknown:
                extra = ", ".join(unknown)
                return f"I can't use {name} with unsupported {extra} information."

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
