"""Policy-aware CAR-bench controller.

The controller handles deterministic safety/disambiguation flows before the
agent falls back to an LLM. It deliberately uses only information visible to
the agent under test: the system/wiki prompt, current tool schema, user turns,
and tool observations.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .tool_index import ToolIndex


BAD_SUNROOF_WEATHER = {"rainy", "cloudy_and_rain", "foggy", "snowy"}
SAFE_SUNROOF_WEATHER = {"sunny", "cloudy", "partly_cloudy"}


@dataclass
class NextAction:
    """One benchmark-visible assistant action."""

    action: Literal["respond", "tool_calls"]
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reason: str = "policy"

    @classmethod
    def respond(cls, content: str, *, reason: str = "policy") -> "NextAction":
        return cls(action="respond", content=content, reason=reason)

    @classmethod
    def tool_call(
        cls,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        reason: str = "policy",
    ) -> "NextAction":
        return cls(
            action="tool_calls",
            content="",
            tool_calls=[{"tool_name": tool_name, "arguments": arguments or {}}],
            reason=reason,
        )


@dataclass
class RuntimeContext:
    location_id: str | None = None
    month: int | None = None
    day: int | None = None
    hour: int | None = None
    minute: int | None = None


@dataclass
class SunroofFlow:
    active: bool = False
    target_percentage: int | None = None
    requested_close: bool = False
    position_checked: bool = False
    weather_checked: bool = False
    preferences_checked: bool = False
    weather_confirmation_requested: bool = False
    weather_confirmed: bool = False
    sunroof_position: float | None = None
    sunshade_position: float | None = None
    weather_condition: str | None = None
    completed: bool = False


@dataclass
class WindowFlow:
    active: bool = False
    window: str = "ALL"
    target_percentage: int | None = None
    climate_checked: bool = False
    ac_on: bool | None = None
    ac_confirmation_requested: bool = False
    ac_confirmed: bool = False
    completed: bool = False


@dataclass
class AmbientLightFlow:
    active: bool = False
    target_color: str | None = None
    on: bool = True
    preferences_checked: bool = False
    completed: bool = False


@dataclass
class TrunkFlow:
    active: bool = False
    action: Literal["OPEN", "CLOSE"] = "OPEN"
    confirmation_requested: bool = False
    confirmed: bool = False
    completed: bool = False


@dataclass
class AirCirculationFlow:
    active: bool = False
    mode: Literal["FRESH_AIR", "RECIRCULATION", "AUTO"] | None = None
    preferences_checked: bool = False
    completed: bool = False


@dataclass
class ControllerState:
    runtime: RuntimeContext = field(default_factory=RuntimeContext)
    sunroof: SunroofFlow = field(default_factory=SunroofFlow)
    window: WindowFlow = field(default_factory=WindowFlow)
    ambient_light: AmbientLightFlow = field(default_factory=AmbientLightFlow)
    trunk: TrunkFlow = field(default_factory=TrunkFlow)
    air_circulation: AirCirculationFlow = field(default_factory=AirCirculationFlow)
    last_user_text: str = ""


class PolicyAwareController:
    """Deterministic guard layer for high-value CAR-bench policies."""

    def __init__(self) -> None:
        self._states: dict[str, ControllerState] = {}

    def reset(self, context_id: str) -> None:
        self._states.pop(context_id, None)

    def decide(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        latest_user_text: str | None = None,
        latest_tool_results: list[dict[str, Any]] | None = None,
    ) -> NextAction | None:
        """Return a policy action, or None to let the model decide."""

        state = self._states.setdefault(context_id, ControllerState())
        tool_index = ToolIndex(tools)
        self._sync_runtime_context(state, messages)

        if latest_user_text:
            self._observe_user_text(state, latest_user_text)

        if latest_tool_results:
            self._observe_tool_results(state, latest_tool_results)

        if state.sunroof.active:
            return self._next_sunroof_action(state, tool_index)
        if state.window.active:
            return self._next_window_action(state, tool_index)
        if state.ambient_light.active:
            return self._next_ambient_light_action(state, tool_index)
        if state.trunk.active:
            return self._next_trunk_action(state, tool_index)
        if state.air_circulation.active:
            return self._next_air_circulation_action(state, tool_index)

        return None

    def _sync_runtime_context(
        self, state: ControllerState, messages: list[dict[str, Any]]
    ) -> None:
        if not messages or messages[0].get("role") != "system":
            return
        system_prompt = messages[0].get("content") or ""

        location = _extract_json_after_label(system_prompt, "CURRENT_LOCATION")
        if isinstance(location, dict):
            state.runtime.location_id = location.get("id") or state.runtime.location_id

        datetime_value = _extract_json_after_label(system_prompt, "DATETIME")
        if isinstance(datetime_value, dict):
            state.runtime.month = _safe_int(datetime_value.get("month"), state.runtime.month)
            state.runtime.day = _safe_int(datetime_value.get("day"), state.runtime.day)
            state.runtime.hour = _safe_int(datetime_value.get("hour"), state.runtime.hour)
            state.runtime.minute = _safe_int(datetime_value.get("minute"), state.runtime.minute)

    def _observe_user_text(self, state: ControllerState, text: str) -> None:
        text = text.strip()
        if not text:
            return
        state.last_user_text = text
        lowered = text.lower()

        if "###stop###" in lowered:
            return

        sunroof = state.sunroof
        if sunroof.weather_confirmation_requested and _is_affirmative(lowered):
            sunroof.weather_confirmed = True
            sunroof.weather_confirmation_requested = False
            return

        window = state.window
        if window.ac_confirmation_requested and _is_affirmative(lowered):
            window.ac_confirmed = True
            window.ac_confirmation_requested = False
            return

        trunk = state.trunk
        if trunk.confirmation_requested and _is_affirmative(lowered):
            trunk.confirmed = True
            trunk.confirmation_requested = False
            return

        if sunroof.active and sunroof.target_percentage is None:
            clarified = _extract_percentage(lowered, allow_standalone=True)
            if clarified is not None:
                sunroof.target_percentage = clarified
                return

        if window.active and window.target_percentage is None:
            clarified = _extract_percentage(lowered, allow_standalone=True)
            if clarified is not None:
                window.target_percentage = clarified
                return

        if state.ambient_light.active and state.ambient_light.target_color is None:
            clarified_color = _extract_ambient_color(lowered)
            if clarified_color is not None:
                state.ambient_light.target_color = clarified_color
                return

        if state.air_circulation.active and state.air_circulation.mode is None:
            clarified_mode = _extract_air_circulation_mode(lowered)
            if clarified_mode is not None:
                state.air_circulation.mode = clarified_mode
                return

        if _is_sunroof_open_request(lowered):
            state.sunroof = SunroofFlow(
                active=True,
                target_percentage=_extract_sunroof_percentage(lowered),
            )
        elif _is_window_open_request(lowered):
            state.window = WindowFlow(
                active=True,
                window=_extract_window_target(lowered),
                target_percentage=_extract_window_percentage(lowered),
            )
        elif _is_ambient_light_request(lowered):
            state.ambient_light = AmbientLightFlow(
                active=True,
                target_color=_extract_ambient_color(lowered),
                on=not _is_light_off_request(lowered),
            )
        elif _is_trunk_request(lowered):
            state.trunk = TrunkFlow(
                active=True,
                action="CLOSE" if _is_close_request(lowered) else "OPEN",
            )
        elif _is_air_circulation_request(lowered):
            state.air_circulation = AirCirculationFlow(
                active=True,
                mode=_extract_air_circulation_mode(lowered),
            )

    def _observe_tool_results(
        self, state: ControllerState, tool_results: list[dict[str, Any]]
    ) -> None:
        for tool_result in tool_results:
            name = _tool_result_name(tool_result)
            payload = _parse_tool_result_content(tool_result)
            result = payload.get("result") if isinstance(payload, dict) else None

            if name == "get_sunroof_and_sunshade_position" and isinstance(result, dict):
                state.sunroof.position_checked = True
                state.sunroof.sunroof_position = _safe_float(
                    result.get("sunroof_position"), state.sunroof.sunroof_position
                )
                state.sunroof.sunshade_position = _safe_float(
                    result.get("sunshade_position"), state.sunroof.sunshade_position
                )
            elif name == "get_weather" and isinstance(result, dict):
                state.sunroof.weather_checked = True
                current_slot = result.get("current_slot") or {}
                condition = current_slot.get("condition")
                if isinstance(condition, str):
                    state.sunroof.weather_condition = condition
            elif name == "get_user_preferences" and isinstance(result, dict):
                state.sunroof.preferences_checked = True
                preferred = _extract_preferred_sunroof_percentage(result)
                if preferred is not None and state.sunroof.target_percentage is None:
                    state.sunroof.target_percentage = preferred

                state.ambient_light.preferences_checked = True
                preferred_color = _extract_preferred_ambient_color(result)
                if (
                    preferred_color is not None
                    and state.ambient_light.target_color is None
                ):
                    state.ambient_light.target_color = preferred_color
                    state.ambient_light.on = preferred_color != "NONE"

                state.air_circulation.preferences_checked = True
                preferred_mode = _extract_preferred_air_circulation_mode(result)
                if preferred_mode is not None and state.air_circulation.mode is None:
                    state.air_circulation.mode = preferred_mode
            elif name == "open_close_sunshade" and isinstance(result, dict):
                state.sunroof.sunshade_position = _safe_float(
                    result.get("percentage"), state.sunroof.sunshade_position
                )
            elif name == "open_close_sunroof" and isinstance(result, dict):
                state.sunroof.sunroof_position = _safe_float(
                    result.get("percentage"), state.sunroof.sunroof_position
                )
                state.sunroof.completed = True
            elif name == "get_climate_settings" and isinstance(result, dict):
                state.window.climate_checked = True
                ac_value = result.get("air_conditioning")
                if isinstance(ac_value, bool):
                    state.window.ac_on = ac_value
            elif name == "open_close_window" and isinstance(result, dict):
                state.window.completed = True
                state.window.target_percentage = _safe_int(
                    result.get("percentage"), state.window.target_percentage
                )
                window = result.get("window")
                if isinstance(window, str):
                    state.window.window = window
            elif name == "set_ambient_lights" and isinstance(result, dict):
                state.ambient_light.completed = True
                color = result.get("lightcolor")
                if isinstance(color, str):
                    state.ambient_light.target_color = color
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.ambient_light.on = on_value
            elif name == "open_close_trunk_door" and isinstance(result, dict):
                state.trunk.completed = True
                action = result.get("action")
                if action in {"OPEN", "CLOSE"}:
                    state.trunk.action = action
            elif name == "set_air_circulation" and isinstance(result, dict):
                state.air_circulation.completed = True
                mode = result.get("mode")
                if mode in {"FRESH_AIR", "RECIRCULATION", "AUTO"}:
                    state.air_circulation.mode = mode

    def _next_sunroof_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        sunroof = state.sunroof

        if sunroof.completed:
            target = _format_percentage(sunroof.sunroof_position)
            shade = _format_percentage(sunroof.sunshade_position)
            return NextAction.respond(
                f"Done, the sunroof is open to {target} and the sunshade is at {shade}.",
                reason="sunroof_done",
            )

        if sunroof.target_percentage is None:
            if not sunroof.preferences_checked and tool_index.has("get_user_preferences"):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {"vehicle_settings": True}
                        }
                    },
                    reason="sunroof_internal_disambiguation_preferences",
                )
            return NextAction.respond(
                "How far would you like me to open the sunroof?",
                reason="sunroof_user_disambiguation",
            )

        if not sunroof.position_checked and tool_index.has(
            "get_sunroof_and_sunshade_position"
        ):
            return NextAction.tool_call(
                "get_sunroof_and_sunshade_position",
                reason="sunroof_precondition_state_check",
            )

        if sunroof.target_percentage > 0 and not sunroof.weather_checked:
            if not tool_index.has("get_weather"):
                return NextAction.respond(
                    "I can't safely open the sunroof because the weather check is unavailable right now.",
                    reason="sunroof_missing_weather_tool",
                )
            weather_args = _weather_arguments(state.runtime)
            if weather_args is None:
                return NextAction.respond(
                    "I need the current location and time before I can safely open the sunroof.",
                    reason="sunroof_missing_weather_context",
                )
            return NextAction.tool_call(
                "get_weather",
                weather_args,
                reason="sunroof_weather_check",
            )

        if (
            sunroof.target_percentage > 0
            and _requires_weather_confirmation(sunroof.weather_condition)
            and not sunroof.weather_confirmed
        ):
            sunroof.weather_confirmation_requested = True
            condition = _friendly_weather(sunroof.weather_condition)
            return NextAction.respond(
                f"It's {condition} right now, so opening the sunroof needs your confirmation. Do you still want me to open it?",
                reason="sunroof_bad_weather_confirmation",
            )

        if sunroof.target_percentage > 0 and (sunroof.sunshade_position or 0) < 100:
            if not tool_index.has("open_close_sunshade"):
                return NextAction.respond(
                    "I can't open the sunroof safely because the sunshade control is unavailable, and the sunshade needs to be fully open first.",
                    reason="sunroof_missing_sunshade_tool",
                )
            return NextAction.tool_call(
                "open_close_sunshade",
                {"percentage": 100},
                reason="sunroof_open_sunshade_first",
            )

        if not tool_index.has("open_close_sunroof"):
            return NextAction.respond(
                "I can't open the sunroof because the sunroof control is unavailable right now.",
                reason="sunroof_missing_sunroof_tool",
            )

        return NextAction.tool_call(
            "open_close_sunroof",
            {"percentage": sunroof.target_percentage},
            reason="sunroof_open",
        )

    def _next_window_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        window = state.window

        if window.completed:
            verb = "are" if window.window == "ALL" else "is"
            return NextAction.respond(
                f"Done, {friendly_window_name(window.window)} {verb} open to {_format_percentage(window.target_percentage)}.",
                reason="window_done",
            )

        if not tool_index.has("open_close_window"):
            return NextAction.respond(
                "I can't move the windows because the window control is unavailable right now.",
                reason="window_missing_tool",
            )

        window_args = tool_index.arg_names("open_close_window")
        if "window" not in window_args or "percentage" not in window_args:
            return NextAction.respond(
                "I can't set the windows safely because the required window position controls are unavailable right now.",
                reason="window_missing_required_parameter",
            )

        if window.target_percentage is None:
            return NextAction.respond(
                "How far should I open the windows?",
                reason="window_user_disambiguation",
            )

        if window.target_percentage > 25 and not window.climate_checked:
            if tool_index.has("get_climate_settings"):
                return NextAction.tool_call(
                    "get_climate_settings",
                    reason="window_ac_policy_check",
                )

        if (
            window.target_percentage > 25
            and window.ac_on is True
            and not window.ac_confirmed
        ):
            window.ac_confirmation_requested = True
            return NextAction.respond(
                "The air conditioning is on, so opening the windows that far can waste energy. Do you still want me to open them?",
                reason="window_ac_confirmation",
            )

        return NextAction.tool_call(
            "open_close_window",
            {"window": window.window, "percentage": window.target_percentage},
            reason="window_open",
        )

    def _next_ambient_light_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        ambient = state.ambient_light

        if ambient.completed:
            if ambient.on:
                return NextAction.respond(
                    f"Done, the ambient lights are set to {ambient.target_color}.",
                    reason="ambient_light_done",
                )
            return NextAction.respond(
                "Done, the ambient lights are off.",
                reason="ambient_light_done",
            )

        if not tool_index.has("set_ambient_lights"):
            return NextAction.respond(
                "I can't change the ambient lights because that control is unavailable right now.",
                reason="ambient_light_missing_tool",
            )

        ambient_args = tool_index.arg_names("set_ambient_lights")
        if "on" not in ambient_args or "lightcolor" not in ambient_args:
            return NextAction.respond(
                "I can't set the ambient light color because that required control is unavailable right now.",
                reason="ambient_light_missing_required_parameter",
            )

        if not ambient.on:
            return NextAction.tool_call(
                "set_ambient_lights",
                {"on": False, "lightcolor": "NONE"},
                reason="ambient_light_off",
            )

        if ambient.target_color is None:
            if not ambient.preferences_checked and tool_index.has("get_user_preferences"):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {"vehicle_settings": True}
                        }
                    },
                    reason="ambient_light_internal_disambiguation_preferences",
                )
            return NextAction.respond(
                "Which ambient light color would you like?",
                reason="ambient_light_user_disambiguation",
            )

        return NextAction.tool_call(
            "set_ambient_lights",
            {"on": True, "lightcolor": ambient.target_color},
            reason="ambient_light_set",
        )

    def _next_trunk_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        trunk = state.trunk

        if trunk.completed:
            return NextAction.respond(
                "Done, the trunk is open."
                if trunk.action == "OPEN"
                else "Done, the trunk is closed.",
                reason="trunk_done",
            )

        if not tool_index.has("open_close_trunk_door"):
            return NextAction.respond(
                "I can't move the trunk door because that control is unavailable right now.",
                reason="trunk_missing_tool",
            )

        if not trunk.confirmed:
            trunk.confirmation_requested = True
            return NextAction.respond(
                "Please confirm if you want me to open the trunk."
                if trunk.action == "OPEN"
                else "Please confirm if you want me to close the trunk.",
                reason="trunk_confirmation",
            )

        return NextAction.tool_call(
            "open_close_trunk_door",
            {"action": trunk.action},
            reason="trunk_action",
        )

    def _next_air_circulation_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        air = state.air_circulation

        if air.completed:
            return NextAction.respond(
                f"Done, air circulation is set to {_friendly_air_mode(air.mode)}.",
                reason="air_circulation_done",
            )

        if not tool_index.has("set_air_circulation"):
            return NextAction.respond(
                "I can't change the air circulation because that control is unavailable right now.",
                reason="air_circulation_missing_tool",
            )

        if "mode" not in tool_index.arg_names("set_air_circulation"):
            return NextAction.respond(
                "I can't change the air circulation mode because that required control is unavailable right now.",
                reason="air_circulation_missing_mode_parameter",
            )

        if air.mode is None:
            if not air.preferences_checked and tool_index.has("get_user_preferences"):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {"climate_control": True}
                        }
                    },
                    reason="air_circulation_internal_disambiguation_preferences",
                )
            return NextAction.respond(
                "Which air circulation mode would you like: fresh air, recirculation, or auto?",
                reason="air_circulation_user_disambiguation",
            )

        return NextAction.tool_call(
            "set_air_circulation",
            {"mode": air.mode},
            reason="air_circulation_set",
        )


def _extract_json_after_label(text: str, label: str) -> dict[str, Any] | None:
    label_index = text.find(label)
    if label_index == -1:
        return None
    start = text.find("{", label_index)
    if start == -1:
        return None
    raw = _balanced_json_object(text, start)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _balanced_json_object(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _is_sunroof_open_request(text: str) -> bool:
    if "sunroof" not in text:
        return False
    if any(word in text for word in ("close", "shut")):
        return False
    return any(word in text for word in ("open", "fresh air", "vent"))


def _is_window_open_request(text: str) -> bool:
    if "sunroof" in text:
        return False
    if not re.search(r"\bwindows?\b", text):
        return False
    if _is_close_request(text):
        return False
    return any(
        phrase in text
        for phrase in (
            "open",
            "fresh air",
            "ventilation",
            "ventilate",
            "stuffy",
            "air out",
            "crack",
            "roll down",
        )
    )


def _is_ambient_light_request(text: str) -> bool:
    return bool(
        ("ambient" in text and ("light" in text or "lighting" in text))
        or "cabin light" in text
        or "mood light" in text
    )


def _is_light_off_request(text: str) -> bool:
    return _is_close_request(text) or "turn off" in text or "switch off" in text


def _is_trunk_request(text: str) -> bool:
    if "trunk" not in text and "boot" not in text:
        return False
    return any(word in text for word in ("open", "close", "shut", "access"))


def _is_air_circulation_request(text: str) -> bool:
    if "air circulation" not in text and "recirculation" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "change",
            "set",
            "switch",
            "turn",
            "fresh air",
            "recirculation",
            "preferred",
            "don't like",
            "do not like",
        )
    )


def _is_close_request(text: str) -> bool:
    return any(word in text for word in ("close", "closed", "shut", "off"))


def _extract_window_target(text: str) -> str:
    if "driver rear" in text or "rear driver" in text or "left rear" in text:
        return "DRIVER_REAR"
    if "passenger rear" in text or "rear passenger" in text or "right rear" in text:
        return "PASSENGER_REAR"
    if "driver" in text and "rear" not in text:
        return "DRIVER"
    if "passenger" in text and "rear" not in text:
        return "PASSENGER"
    return "ALL"


def _extract_window_percentage(text: str) -> int | None:
    return _extract_percentage(text)


def _extract_sunroof_percentage(
    text: str, *, allow_standalone: bool = False
) -> int | None:
    if any(word in text for word in ("half", "halfway")):
        return 50

    if re.search(r"\b50\s*%?", text):
        return 50

    if _sunroof_full_requested(text):
        return 100

    if allow_standalone:
        if re.search(r"\b100\s*%?", text) or "fully" in text or "all the way" in text:
            return 100
        match = re.search(r"\b([1-9][0-9]?|100)\s*%?\b", text)
        if match:
            return int(match.group(1))

    sunroof_clause = re.search(r"sunroof[^.?!,;]*?([1-9][0-9]?|100)\s*%?", text)
    if sunroof_clause:
        return int(sunroof_clause.group(1))

    return None


def _extract_percentage(text: str, *, allow_standalone: bool = False) -> int | None:
    if any(word in text for word in ("half", "halfway")):
        return 50
    if re.search(r"\b50\s*%?", text):
        return 50
    if any(phrase in text for phrase in ("fully", "full", "all the way")):
        return 100

    percent_match = re.search(r"\b([1-9][0-9]?|100)\s*%", text)
    if percent_match:
        return int(percent_match.group(1))

    if allow_standalone:
        number_match = re.search(r"\b([1-9][0-9]?|100)\b", text)
        if number_match:
            return int(number_match.group(1))

    return None


def _sunroof_full_requested(text: str) -> bool:
    # Avoid treating "open the sunshade all the way" as a sunroof target.
    if "sunshade" in text and re.search(r"sunshade[^.?!,;]*(fully|all the way|100)", text):
        return False
    return bool(
        re.search(r"sunroof[^.?!,;]*(fully|all the way|100\s*%)", text)
        or re.search(r"(fully|all the way)[^.?!,;]*sunroof", text)
    )


AMBIENT_COLORS = {
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
    "WHITE",
    "PINK",
    "ORANGE",
    "PURPLE",
    "CYAN",
    "NONE",
}


def _extract_ambient_color(text: str) -> str | None:
    upper = text.upper()
    for color in sorted(AMBIENT_COLORS - {"NONE"}, key=len, reverse=True):
        if re.search(rf"\b{re.escape(color)}\b", upper):
            return color
    if "off" in text or "none" in text:
        return "NONE"
    return None


def _extract_air_circulation_mode(
    text: str,
) -> Literal["FRESH_AIR", "RECIRCULATION", "AUTO"] | None:
    if "fresh air" in text or "outside air" in text:
        return "FRESH_AIR"
    if "recirculation" in text or "recirculate" in text:
        if any(phrase in text for phrase in ("don't like", "do not like", "not recirculation")):
            return None
        return "RECIRCULATION"
    if re.search(r"\bauto(?:matic)?\b", text):
        return "AUTO"
    return None


def _is_affirmative(text: str) -> bool:
    text = text.strip().lower()
    return bool(
        re.search(r"\b(yes|yeah|yep|sure|ok|okay|confirm|confirmed)\b", text)
        or "go ahead" in text
        or "still want" in text
        or "proceed" in text
        or "do it" in text
    )


def _tool_result_name(tool_result: dict[str, Any]) -> str:
    return (
        tool_result.get("tool_name")
        or tool_result.get("toolName")
        or tool_result.get("name")
        or ""
    )


def _parse_tool_result_content(tool_result: dict[str, Any]) -> dict[str, Any]:
    content = tool_result.get("content", "")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_preferred_sunroof_percentage(result: dict[str, Any]) -> int | None:
    text = json.dumps(result, ensure_ascii=False).lower()
    match = re.search(r"default value to open the sunroof is\s*(\d{1,3})\s*%", text)
    if match:
        return _bounded_percentage(match.group(1))

    match = re.search(r"sunroof[^.]*?(\d{1,3})\s*%", text)
    if match:
        value = _bounded_percentage(match.group(1))
        if value is not None and not ("never" in text and value == 100):
            return value

    return None


def _extract_preferred_ambient_color(result: dict[str, Any]) -> str | None:
    text = json.dumps(result, ensure_ascii=False).upper()
    explicit = re.search(
        r"(?:LIGHTCOLOR|AMBIENT[^\"']*LIGHT|AMBIENT[^\"']*COLOR)[^A-Z0-9]+"
        r"(RED|GREEN|BLUE|YELLOW|WHITE|PINK|ORANGE|PURPLE|CYAN|NONE)\b",
        text,
    )
    if explicit:
        return explicit.group(1)

    for color in sorted(AMBIENT_COLORS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(color)}\b", text):
            return color
    return None


def _extract_preferred_air_circulation_mode(
    result: dict[str, Any],
) -> Literal["FRESH_AIR", "RECIRCULATION", "AUTO"] | None:
    text = json.dumps(result, ensure_ascii=False).lower()
    if "fresh air" in text or "outside air" in text:
        return "FRESH_AIR"
    if "recirculation" in text or "recirculate" in text:
        return "RECIRCULATION"
    if re.search(r"\bauto(?:matic)?\b", text):
        return "AUTO"
    return None


def friendly_window_name(window: str) -> str:
    return {
        "ALL": "all windows",
        "DRIVER": "the driver window",
        "PASSENGER": "the passenger window",
        "DRIVER_REAR": "the driver rear window",
        "PASSENGER_REAR": "the passenger rear window",
        "RIGHT_REAR": "the right rear window",
        "LEFT_REAR": "the left rear window",
    }.get(window, "the window")


def _friendly_air_mode(
    mode: Literal["FRESH_AIR", "RECIRCULATION", "AUTO"] | None,
) -> str:
    return {
        "FRESH_AIR": "fresh air mode",
        "RECIRCULATION": "recirculation mode",
        "AUTO": "auto mode",
        None: "the requested mode",
    }[mode]


def _bounded_percentage(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(0, min(100, number))


def _weather_arguments(runtime: RuntimeContext) -> dict[str, Any] | None:
    if (
        runtime.location_id is None
        or runtime.month is None
        or runtime.day is None
        or runtime.hour is None
    ):
        return None
    args: dict[str, Any] = {
        "location_or_poi_id": runtime.location_id,
        "month": runtime.month,
        "day": runtime.day,
        "time_hour_24hformat": runtime.hour,
    }
    if runtime.minute is not None:
        args["time_minutes"] = runtime.minute
    return args


def _requires_weather_confirmation(condition: str | None) -> bool:
    if not condition:
        return True
    return condition not in SAFE_SUNROOF_WEATHER


def _friendly_weather(condition: str | None) -> str:
    if not condition:
        return "not in a confirmed safe weather condition"
    return condition.replace("_", " ")


def _format_percentage(value: float | int | None) -> str:
    if value is None:
        return "the requested position"
    if float(value).is_integer():
        return f"{int(value)}%"
    return f"{value}%"


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
