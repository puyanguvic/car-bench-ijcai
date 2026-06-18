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

from .response_renderer import (
    clean_user_content,
    render_malformed_tool_arguments,
    render_malformed_tool_call,
)
from .tool_index import ToolIndex


BAD_SUNROOF_WEATHER = {"rainy", "cloudy_and_rain", "foggy", "snowy"}
SAFE_SUNROOF_WEATHER = {"sunny", "cloudy", "partly_cloudy"}
SAFE_FOG_LIGHT_WEATHER = {"cloudy_and_thunderstorm", "cloudy_and_hail"}


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
class SunshadeFlow:
    active: bool = False
    target_percentage: int | None = None
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
class WindowMatchFlow:
    active: bool = False
    windows_checked: bool = False
    window_driver_position: int | None = None
    window_passenger_position: int | None = None
    window_driver_rear_position: int | None = None
    window_passenger_rear_position: int | None = None
    completed: bool = False


@dataclass
class AmbientLightFlow:
    active: bool = False
    target_color: str | None = None
    on: bool = True
    match_car_color: bool = False
    car_color_checked: bool = False
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
class AirConditioningFlow:
    active: bool = False
    on: bool = True
    fan_target_level: int | None = None
    climate_checked: bool = False
    windows_checked: bool = False
    fan_speed: int | None = None
    air_conditioning: bool | None = None
    window_driver_position: int | None = None
    window_passenger_position: int | None = None
    window_driver_rear_position: int | None = None
    window_passenger_rear_position: int | None = None
    completed: bool = False


@dataclass
class AirQualityFlow:
    active: bool = False
    climate_checked: bool = False
    fan_speed: int | None = None


@dataclass
class FanSpeedFlow:
    active: bool = False
    level: int | None = None
    preferences_checked: bool = False
    completed: bool = False


@dataclass
class SteeringWheelHeatingFlow:
    active: bool = False
    level: int | None = None
    preferences_checked: bool = False
    completed: bool = False


@dataclass
class ReadingLightFlow:
    active: bool = False
    position: str | None = None
    on: bool = True
    completed: bool = False


@dataclass
class ReadingLightOccupancyFlow:
    active: bool = False
    seats_checked: bool = False
    seats_occupied: dict[str, bool] = field(default_factory=dict)
    pending_actions: list[tuple[str, bool]] = field(default_factory=list)
    completed: bool = False


@dataclass
class NavigationFlow:
    active: bool = False
    mode: (
        Literal[
            "set_new",
            "replace_final_destination",
            "delete_final_destination",
            "delete_waypoint",
            "replace_one_waypoint",
        ]
        | None
    ) = None
    destination_name: str | None = None
    destination_id: str | None = None
    route_start_id: str | None = None
    route_preference: Literal["fastest", "shortest"] | None = None
    routes_checked: bool = False
    routes: list[dict[str, Any]] = field(default_factory=list)
    route_choice_requested: bool = False
    selected_route_index: int | None = None
    current_navigation_checked: bool = False
    needs_current_navigation: bool = False
    navigation_active: bool | None = None
    waypoints_id: list[str] = field(default_factory=list)
    routes_to_final_destination_id: list[str] = field(default_factory=list)
    waypoint_details: list[dict[str, Any]] = field(default_factory=list)
    waypoint_name: str | None = None
    waypoint_id: str | None = None
    new_waypoint_name: str | None = None
    new_waypoint_id: str | None = None
    route_lookup: (
        Literal[
            "destination",
            "delete_without_waypoint",
            "to_new_waypoint",
            "from_new_waypoint",
        ]
        | None
    ) = None
    route_without_waypoint_checked: bool = False
    routes_without_waypoint: list[dict[str, Any]] = field(default_factory=list)
    route_to_new_waypoint_checked: bool = False
    routes_to_new_waypoint: list[dict[str, Any]] = field(default_factory=list)
    route_from_new_waypoint_checked: bool = False
    routes_from_new_waypoint: list[dict[str, Any]] = field(default_factory=list)
    completion_message: str | None = None
    completed: bool = False
    failure_message: str | None = None


@dataclass
class EmailFlow:
    active: bool = False
    mode: Literal["meeting_delay", "share_contact"] | None = None
    recipient_name: str | None = None
    recipient_first_name: str | None = None
    recipient_last_name: str | None = None
    recipient_contact_id: str | None = None
    recipient_email: str | None = None
    subject_name: str | None = None
    subject_first_name: str | None = None
    subject_last_name: str | None = None
    subject_contact_id: str | None = None
    subject_email: str | None = None
    subject_phone: str | None = None
    calendar_checked: bool = False
    meeting_topic: str | None = None
    meeting_started: bool | None = None
    meeting_start_hour: int | None = None
    meeting_start_minute: int | None = None
    user_claimed_late: bool = False
    pending_lookup_role: Literal["recipient", "subject"] | None = None
    pending_contact_matches: dict[str, str] = field(default_factory=dict)
    preferences_checked: bool = False
    content_message: str | None = None
    confirmation_requested: bool = False
    confirmed: bool = False
    completed: bool = False
    failure_message: str | None = None


@dataclass
class HighBeamFlow:
    active: bool = False
    on: bool = True
    exterior_lights_checked: bool = False
    fog_lights_on: bool | None = None
    confirmation_requested: bool = False
    confirmed: bool = False
    declined: bool = False
    completed: bool = False


@dataclass
class FogLightFlow:
    active: bool = False
    on: bool = True
    weather_checked: bool = False
    weather_condition: str | None = None
    weather_confirmation_requested: bool = False
    weather_confirmed: bool = False
    exterior_lights_checked: bool = False
    fog_lights_on: bool | None = None
    low_beams_on: bool | None = None
    high_beams_on: bool | None = None
    high_beam_confirmation_requested: bool = False
    high_beam_confirmed: bool = False
    declined: bool = False
    completed: bool = False


@dataclass
class DefrostFlow:
    active: bool = False
    on: bool = True
    defrost_window: Literal["ALL", "FRONT", "REAR"] | None = None
    climate_checked: bool = False
    windows_checked: bool = False
    fan_speed: int | None = None
    fan_airflow_direction: str | None = None
    air_conditioning: bool | None = None
    window_driver_position: int | None = None
    window_passenger_position: int | None = None
    window_driver_rear_position: int | None = None
    window_passenger_rear_position: int | None = None
    completed: bool = False


@dataclass
class ControllerState:
    runtime: RuntimeContext = field(default_factory=RuntimeContext)
    sunroof: SunroofFlow = field(default_factory=SunroofFlow)
    sunshade: SunshadeFlow = field(default_factory=SunshadeFlow)
    window: WindowFlow = field(default_factory=WindowFlow)
    window_match: WindowMatchFlow = field(default_factory=WindowMatchFlow)
    ambient_light: AmbientLightFlow = field(default_factory=AmbientLightFlow)
    trunk: TrunkFlow = field(default_factory=TrunkFlow)
    air_circulation: AirCirculationFlow = field(default_factory=AirCirculationFlow)
    air_conditioning: AirConditioningFlow = field(default_factory=AirConditioningFlow)
    air_quality: AirQualityFlow = field(default_factory=AirQualityFlow)
    fan_speed: FanSpeedFlow = field(default_factory=FanSpeedFlow)
    steering_wheel_heating: SteeringWheelHeatingFlow = field(
        default_factory=SteeringWheelHeatingFlow
    )
    reading_light: ReadingLightFlow = field(default_factory=ReadingLightFlow)
    reading_light_occupancy: ReadingLightOccupancyFlow = field(
        default_factory=ReadingLightOccupancyFlow
    )
    navigation: NavigationFlow = field(default_factory=NavigationFlow)
    email: EmailFlow = field(default_factory=EmailFlow)
    high_beam: HighBeamFlow = field(default_factory=HighBeamFlow)
    fog_lights: FogLightFlow = field(default_factory=FogLightFlow)
    defrost: DefrostFlow = field(default_factory=DefrostFlow)
    last_user_text: str = ""
    recent_meeting_topic: str | None = None
    recent_calendar_meetings: list[dict[str, Any]] = field(default_factory=list)
    business_email_extra_recipients: list[str] = field(default_factory=list)
    pending_location_lookup_name: str | None = None
    recent_location_lookup_name: str | None = None
    recent_location_lookup_id: str | None = None


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
        self._sync_recent_tool_call_arguments(state, messages)

        if latest_user_text:
            self._observe_user_text(state, latest_user_text)

        if latest_tool_results:
            self._observe_tool_results(state, latest_tool_results)

        action: NextAction | None = None
        if state.sunroof.active:
            action = self._next_sunroof_action(state, tool_index)
        elif state.sunshade.active:
            action = self._next_sunshade_action(state, tool_index)
        elif state.window_match.active:
            action = self._next_window_match_action(state, tool_index)
        elif state.window.active:
            action = self._next_window_action(state, tool_index)
        elif state.ambient_light.active:
            action = self._next_ambient_light_action(state, tool_index)
        elif state.trunk.active:
            action = self._next_trunk_action(state, tool_index)
        elif state.air_circulation.active:
            action = self._next_air_circulation_action(state, tool_index)
        elif state.air_conditioning.active:
            action = self._next_air_conditioning_action(state, tool_index)
        elif state.air_quality.active:
            action = self._next_air_quality_action(state, tool_index)
        elif state.fan_speed.active:
            action = self._next_fan_speed_action(state, tool_index)
        elif state.steering_wheel_heating.active:
            action = self._next_steering_wheel_heating_action(state, tool_index)
        elif state.reading_light_occupancy.active:
            action = self._next_reading_light_occupancy_action(state, tool_index)
        elif state.reading_light.active:
            action = self._next_reading_light_action(state, tool_index)
        elif state.defrost.active:
            action = self._next_defrost_action(state, tool_index)
        elif state.navigation.active:
            action = self._next_navigation_action(state, tool_index)
        elif state.email.active:
            action = self._next_email_action(state, tool_index)
        elif state.high_beam.active:
            action = self._next_high_beam_action(state, tool_index)
        elif state.fog_lights.active:
            action = self._next_fog_lights_action(state, tool_index)

        if action is not None:
            return self._validated_controller_action(action, tool_index)

        return None

    def _validated_controller_action(
        self, action: NextAction, tool_index: ToolIndex
    ) -> NextAction:
        if action.action == "respond":
            return NextAction.respond(
                clean_user_content(action.content),
                reason=action.reason,
            )

        if action.action != "tool_calls" or not action.tool_calls:
            return NextAction.respond(
                render_malformed_tool_call(),
                reason=f"{action.reason}_invalid_tool_call",
            )

        for tool_call in action.tool_calls:
            if not isinstance(tool_call, dict):
                return NextAction.respond(
                    render_malformed_tool_call(),
                    reason=f"{action.reason}_invalid_tool_call",
                )
            tool_name = tool_call.get("tool_name")
            arguments = tool_call.get("arguments") or {}
            if not isinstance(tool_name, str) or not tool_name:
                return NextAction.respond(
                    render_malformed_tool_call(),
                    reason=f"{action.reason}_invalid_tool_call",
                )
            if not isinstance(arguments, dict):
                return NextAction.respond(
                    render_malformed_tool_arguments(),
                    reason=f"{action.reason}_invalid_tool_arguments",
                )
            validation_error = tool_index.validate_call(tool_name, arguments)
            if validation_error:
                return NextAction.respond(
                    validation_error,
                    reason=f"{action.reason}_schema_guard",
                )

        return action

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
            state.runtime.month = _safe_int(
                datetime_value.get("month"), state.runtime.month
            )
            state.runtime.day = _safe_int(datetime_value.get("day"), state.runtime.day)
            state.runtime.hour = _safe_int(
                datetime_value.get("hour"), state.runtime.hour
            )
            state.runtime.minute = _safe_int(
                datetime_value.get("minute"), state.runtime.minute
            )

        business_email_extra_recipients = _extract_business_email_extra_recipients(
            system_prompt
        )
        if business_email_extra_recipients:
            state.business_email_extra_recipients = business_email_extra_recipients

    def _sync_recent_tool_call_arguments(
        self, state: ControllerState, messages: list[dict[str, Any]]
    ) -> None:
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            if not message.get("tool_calls"):
                return
            for tool_call in reversed(message["tool_calls"]):
                function = tool_call.get("function") or {}
                if function.get("name") != "get_location_id_by_location_name":
                    continue
                arguments = _parse_tool_call_arguments(function.get("arguments"))
                location = arguments.get("location")
                if isinstance(location, str):
                    state.pending_location_lookup_name = (
                        _clean_location_query(location) or location.strip()
                    )
                return
            return

    def _observe_user_text(self, state: ControllerState, text: str) -> None:
        text = text.strip()
        if not text:
            return
        previous_user_text = state.last_user_text
        state.last_user_text = text
        lowered = text.lower()

        if "###stop###" in lowered:
            return

        quoted_meeting = _extract_quoted_meeting_topic(text)
        if quoted_meeting is not None:
            state.recent_meeting_topic = quoted_meeting

        sunroof = state.sunroof
        if sunroof.weather_confirmation_requested and _is_affirmative(lowered):
            sunroof.weather_confirmed = True
            sunroof.weather_confirmation_requested = False
            return

        email = state.email
        if email.confirmation_requested:
            if _is_affirmative(lowered):
                email.confirmed = True
                email.confirmation_requested = False
                return
            if _is_negative(lowered):
                state.email = EmailFlow()
                return

        if email.active and email.pending_contact_matches:
            selected = _select_contact_match_from_text(
                email.pending_contact_matches, text
            )
            if selected is not None:
                if email.pending_lookup_role == "recipient":
                    email.recipient_contact_id = selected
                    email.recipient_name = email.pending_contact_matches[selected]
                    _set_email_flow_name(email, "recipient", email.recipient_name)
                elif email.pending_lookup_role == "subject":
                    email.subject_contact_id = selected
                    email.subject_name = email.pending_contact_matches[selected]
                    _set_email_flow_name(email, "subject", email.subject_name)
                email.pending_contact_matches = {}
                email.pending_lookup_role = None
                return

        if email.active and email.mode == "share_contact":
            subject_name = _extract_contact_share_subject(text)
            if subject_name is not None and email.subject_contact_id is None:
                email.subject_name = subject_name
                _set_email_flow_name(email, "subject", subject_name)
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

        high_beam = state.high_beam
        if high_beam.confirmation_requested:
            if _is_affirmative(lowered):
                high_beam.confirmed = True
                high_beam.confirmation_requested = False
                return
            if _is_negative(lowered):
                high_beam.declined = True
                high_beam.confirmation_requested = False
                return

        fog_lights = state.fog_lights
        if fog_lights.weather_confirmation_requested:
            if _is_affirmative(lowered):
                fog_lights.weather_confirmed = True
                fog_lights.weather_confirmation_requested = False
                return
            if _is_negative(lowered):
                fog_lights.declined = True
                fog_lights.weather_confirmation_requested = False
                return

        if fog_lights.high_beam_confirmation_requested:
            if _is_affirmative(lowered):
                fog_lights.high_beam_confirmed = True
                fog_lights.high_beam_confirmation_requested = False
                return
            if _is_negative(lowered):
                fog_lights.declined = True
                fog_lights.high_beam_confirmation_requested = False
                return

        if sunroof.active and sunroof.target_percentage is None:
            clarified = _extract_percentage(lowered, allow_standalone=True)
            if clarified is not None:
                sunroof.target_percentage = clarified
                return

        if state.sunshade.active and state.sunshade.target_percentage is None:
            clarified = _extract_percentage(lowered, allow_standalone=True)
            if clarified is not None:
                state.sunshade.target_percentage = clarified
                return

        if window.active and window.target_percentage is None:
            clarified = _extract_percentage(lowered, allow_standalone=True)
            if clarified is not None:
                window.target_percentage = clarified
                return

        if state.ambient_light.active and state.ambient_light.target_color is None:
            if _is_ambient_light_match_car_color_request(lowered):
                state.ambient_light.match_car_color = True
                return
            clarified_color = _extract_ambient_color(lowered)
            if clarified_color is not None:
                state.ambient_light.target_color = clarified_color
                return

        if state.air_circulation.active and state.air_circulation.mode is None:
            clarified_mode = _extract_air_circulation_mode(lowered)
            if clarified_mode is not None:
                state.air_circulation.mode = clarified_mode
                return

        if state.air_quality.active:
            clarified_level = _extract_level(lowered)
            clarified_mode = _extract_air_circulation_mode(lowered)
            if clarified_level is not None or "fan" in lowered:
                state.fan_speed = FanSpeedFlow(active=True, level=clarified_level)
                state.air_quality = AirQualityFlow()
                return
            if clarified_mode is not None:
                state.air_circulation = AirCirculationFlow(
                    active=True,
                    mode=clarified_mode,
                )
                state.air_quality = AirQualityFlow()
                return
            if _is_air_conditioning_request(lowered):
                state.air_conditioning = AirConditioningFlow(
                    active=True,
                    on=not _is_air_conditioning_off_request(lowered),
                    fan_target_level=_extract_level(lowered),
                )
                state.air_quality = AirQualityFlow()
                return

        if state.fan_speed.active and state.fan_speed.level is None:
            clarified_level = _extract_level(lowered)
            if clarified_level is not None:
                state.fan_speed.level = clarified_level
                return

        if (
            state.steering_wheel_heating.active
            and state.steering_wheel_heating.level is None
        ):
            clarified_level = _extract_heating_level(lowered)
            if clarified_level is not None:
                state.steering_wheel_heating.level = clarified_level
                return

        if state.reading_light.active and state.reading_light.position is None:
            clarified_position = _extract_reading_light_position(lowered)
            if clarified_position is not None:
                state.reading_light.position = clarified_position
                return

        if state.defrost.active and state.defrost.defrost_window is None:
            clarified_window = _extract_defrost_window(lowered)
            if clarified_window is not None:
                state.defrost.defrost_window = clarified_window
                return

        navigation = state.navigation
        if (
            navigation.active
            and navigation.destination_name is None
            and navigation.mode in {"set_new", "replace_final_destination"}
        ):
            clarified_destination = _extract_navigation_clarified_destination(text)
            if clarified_destination is not None:
                navigation.destination_name = clarified_destination
                return

        if navigation.active and navigation.route_choice_requested:
            route_index = _extract_route_choice_index(lowered)
            if route_index is not None:
                navigation.selected_route_index = route_index
                navigation.route_choice_requested = False
                return
            route_preference = _extract_route_preference(lowered)
            if route_preference is not None:
                navigation.route_preference = route_preference
                navigation.route_choice_requested = False
                return
            choice_routes = _navigation_choice_routes(navigation)
            if _is_affirmative(lowered) and _route_choice_is_unambiguous(choice_routes):
                navigation.route_preference = "fastest"
                navigation.route_choice_requested = False
                return

        if _is_sunroof_open_request(lowered):
            state.sunroof = SunroofFlow(
                active=True,
                target_percentage=_extract_sunroof_percentage(lowered),
            )
        elif _is_sunshade_request(lowered):
            state.sunshade = SunshadeFlow(
                active=True,
                target_percentage=_extract_sunshade_percentage(lowered),
            )
        elif _is_window_match_request(lowered):
            state.window_match = WindowMatchFlow(active=True)
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
                match_car_color=_is_ambient_light_match_car_color_request(lowered),
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
        elif _is_air_conditioning_request(lowered):
            state.air_conditioning = AirConditioningFlow(
                active=True,
                on=not _is_air_conditioning_off_request(lowered),
                fan_target_level=_extract_level(lowered),
            )
        elif _is_air_quality_question(lowered):
            state.air_quality = AirQualityFlow(active=True)
        elif _is_fan_speed_request(lowered):
            state.fan_speed = FanSpeedFlow(
                active=True,
                level=0 if _is_light_off_request(lowered) else _extract_level(lowered),
            )
        elif _is_steering_wheel_heating_request(lowered):
            state.steering_wheel_heating = SteeringWheelHeatingFlow(
                active=True,
                level=(
                    0
                    if _is_light_off_request(lowered)
                    else _extract_heating_level(lowered)
                ),
            )
        elif _is_reading_light_by_occupancy_request(lowered):
            state.reading_light_occupancy = ReadingLightOccupancyFlow(active=True)
        elif _is_reading_light_request(lowered):
            state.reading_light = ReadingLightFlow(
                active=True,
                position=_extract_reading_light_position(lowered),
                on=not _is_light_off_request(lowered),
            )
        elif _is_defrost_request(lowered):
            state.defrost = DefrostFlow(
                active=True,
                on=not _is_light_off_request(lowered),
                defrost_window=_extract_defrost_window(lowered),
            )
        elif _is_navigation_delete_destination_request(lowered):
            previous_navigation = state.navigation
            destination_name = _extract_navigation_destination_to_delete(text)
            destination_id = _matching_recent_location_id(
                destination_name,
                previous_navigation.destination_name,
                previous_navigation.destination_id,
            ) or _matching_recent_location_id(
                destination_name,
                state.recent_location_lookup_name,
                state.recent_location_lookup_id,
            )
            state.navigation = NavigationFlow(
                active=True,
                mode="delete_final_destination",
                destination_name=destination_name,
                destination_id=destination_id,
                needs_current_navigation=destination_id is None,
            )
        elif _is_navigation_replace_waypoint_request(lowered):
            state.navigation = NavigationFlow(
                active=True,
                mode="replace_one_waypoint",
                new_waypoint_name=_extract_navigation_new_waypoint(text),
                waypoint_name=_extract_navigation_waypoint_to_replace(text),
                route_preference=_extract_route_preference(lowered),
                needs_current_navigation=True,
            )
        elif _is_navigation_delete_waypoint_request(lowered):
            state.navigation = NavigationFlow(
                active=True,
                mode="delete_waypoint",
                waypoint_name=_extract_navigation_waypoint_to_delete(text),
                route_preference=_extract_route_preference(lowered),
                needs_current_navigation=True,
            )
        elif _is_vague_navigation_destination_request(lowered):
            state.navigation = NavigationFlow(
                active=True,
                mode=(
                    "replace_final_destination"
                    if _is_replace_destination_request(lowered)
                    else "set_new"
                ),
                route_preference=_extract_route_preference(lowered),
                needs_current_navigation=_is_replace_destination_request(lowered),
            )
        elif _is_simple_navigation_request(text):
            destination = _extract_navigation_destination(text)
            if destination is not None:
                previous_navigation = state.navigation
                replace_followup = _is_navigation_destination_change_followup(
                    previous_user_text, lowered
                )
                replace_destination = (
                    _is_replace_destination_request(lowered) or replace_followup
                )
                current_waypoints = (
                    list(previous_navigation.waypoints_id)
                    if replace_destination and previous_navigation.waypoints_id
                    else []
                )
                state.navigation = NavigationFlow(
                    active=True,
                    mode="replace_final_destination"
                    if replace_destination
                    else "set_new",
                    destination_name=destination,
                    route_preference=_extract_route_preference(lowered),
                    needs_current_navigation=(
                        _navigation_replace_needs_current_state(lowered)
                        or bool(current_waypoints)
                        or (
                            replace_followup
                            and ("route" in lowered or "instead" in lowered)
                        )
                    ),
                    current_navigation_checked=bool(current_waypoints),
                    navigation_active=(
                        True
                        if current_waypoints
                        else previous_navigation.navigation_active
                    ),
                    waypoints_id=current_waypoints,
                    routes_to_final_destination_id=list(
                        previous_navigation.routes_to_final_destination_id
                    )
                    if current_waypoints
                    else [],
                )
        elif _is_meeting_delay_email_request(lowered) or (
            state.recent_meeting_topic and _is_late_email_request(lowered)
        ):
            recipient = _extract_email_recipient_name(text)
            if recipient is not None:
                state.email = EmailFlow(
                    active=True,
                    mode="meeting_delay",
                    recipient_name=recipient,
                    meeting_topic=state.recent_meeting_topic,
                    calendar_checked=bool(state.recent_calendar_meetings),
                    user_claimed_late=True,
                )
                _set_email_flow_name(state.email, "recipient", recipient)
        elif _is_contact_share_email_start(lowered):
            recipient = _extract_email_recipient_name(text)
            if recipient is not None:
                state.email = EmailFlow(
                    active=True,
                    mode="share_contact",
                    recipient_name=recipient,
                )
                _set_email_flow_name(state.email, "recipient", recipient)
        elif _is_fog_light_request(lowered):
            state.fog_lights = FogLightFlow(
                active=True,
                on=not _is_light_off_request(lowered),
            )
        elif _is_high_beam_request(lowered):
            state.high_beam = HighBeamFlow(
                active=True,
                on=not _is_light_off_request(lowered),
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
                    state.fog_lights.weather_condition = condition
                if state.fog_lights.active:
                    state.fog_lights.weather_checked = True
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

                state.fan_speed.preferences_checked = True
                preferred_fan_level = _extract_preferred_fan_speed_level(result)
                if preferred_fan_level is not None and state.fan_speed.level is None:
                    state.fan_speed.level = preferred_fan_level

                state.steering_wheel_heating.preferences_checked = True
                preferred_heating_level = _extract_preferred_steering_heating_level(
                    result
                )
                if (
                    preferred_heating_level is not None
                    and state.steering_wheel_heating.level is None
                ):
                    state.steering_wheel_heating.level = preferred_heating_level
                state.business_email_extra_recipients = (
                    state.business_email_extra_recipients
                    + [
                        email
                        for email in _extract_business_email_extra_recipients(
                            json.dumps(result, ensure_ascii=False)
                        )
                        if email not in state.business_email_extra_recipients
                    ]
                )
                if state.email.active:
                    state.email.preferences_checked = True
            elif name == "get_car_color" and isinstance(result, dict):
                if state.ambient_light.active:
                    state.ambient_light.car_color_checked = True
                    car_color = result.get("car_color")
                    if isinstance(car_color, str):
                        color = car_color.upper()
                        if color in AMBIENT_COLORS:
                            state.ambient_light.target_color = color
            elif name == "open_close_sunshade" and isinstance(result, dict):
                state.sunroof.sunshade_position = _safe_float(
                    result.get("percentage"), state.sunroof.sunshade_position
                )
                if state.sunshade.active:
                    state.sunshade.completed = True
                    state.sunshade.target_percentage = _safe_int(
                        result.get("percentage"), state.sunshade.target_percentage
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
                    state.air_conditioning.air_conditioning = ac_value

                state.defrost.climate_checked = True
                fan_speed = _safe_int(result.get("fan_speed"), state.defrost.fan_speed)
                if fan_speed is not None:
                    state.defrost.fan_speed = fan_speed
                    state.air_conditioning.fan_speed = fan_speed
                airflow = result.get("fan_airflow_direction")
                if isinstance(airflow, str):
                    state.defrost.fan_airflow_direction = airflow
                if isinstance(ac_value, bool):
                    state.defrost.air_conditioning = ac_value
                if state.air_conditioning.active:
                    state.air_conditioning.climate_checked = True
                if state.air_quality.active:
                    state.air_quality.climate_checked = True
                    state.air_quality.fan_speed = fan_speed
            elif name == "open_close_window" and isinstance(result, dict):
                state.window.completed = True
                state.window.target_percentage = _safe_int(
                    result.get("percentage"), state.window.target_percentage
                )
                window = result.get("window")
                if isinstance(window, str):
                    state.window.window = window
                    percentage = _safe_int(result.get("percentage"))
                    if percentage is not None:
                        _record_defrost_window_position(
                            state.defrost, window, percentage
                        )
                        _record_air_conditioning_window_position(
                            state.air_conditioning, window, percentage
                        )
                        _record_window_match_position(
                            state.window_match, window, percentage
                        )
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
            elif name == "get_exterior_lights_status" and isinstance(result, dict):
                state.high_beam.exterior_lights_checked = True
                fog_lights = result.get("fog_lights")
                if isinstance(fog_lights, bool):
                    state.high_beam.fog_lights_on = fog_lights
                    state.fog_lights.fog_lights_on = fog_lights
                if state.fog_lights.active:
                    state.fog_lights.exterior_lights_checked = True
                    low_beams = result.get("head_lights_low_beams")
                    if isinstance(low_beams, bool):
                        state.fog_lights.low_beams_on = low_beams
                    high_beams = result.get("head_lights_high_beams")
                    if isinstance(high_beams, bool):
                        state.fog_lights.high_beams_on = high_beams
            elif name == "set_head_lights_low_beams" and isinstance(result, dict):
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.fog_lights.low_beams_on = on_value
            elif name == "set_head_lights_high_beams" and isinstance(result, dict):
                state.high_beam.completed = True
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.high_beam.on = on_value
                    state.fog_lights.high_beams_on = on_value
            elif name == "set_fog_lights" and isinstance(result, dict):
                state.fog_lights.completed = True
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.fog_lights.on = on_value
            elif name == "get_vehicle_window_positions" and isinstance(result, dict):
                state.defrost.windows_checked = True
                state.defrost.window_driver_position = _safe_int(
                    result.get("window_driver_position"),
                    state.defrost.window_driver_position,
                )
                state.window_match.window_driver_position = _safe_int(
                    result.get("window_driver_position"),
                    state.window_match.window_driver_position,
                )
                state.air_conditioning.window_driver_position = _safe_int(
                    result.get("window_driver_position"),
                    state.air_conditioning.window_driver_position,
                )
                state.defrost.window_passenger_position = _safe_int(
                    result.get("window_passenger_position"),
                    state.defrost.window_passenger_position,
                )
                state.window_match.window_passenger_position = _safe_int(
                    result.get("window_passenger_position"),
                    state.window_match.window_passenger_position,
                )
                state.air_conditioning.window_passenger_position = _safe_int(
                    result.get("window_passenger_position"),
                    state.air_conditioning.window_passenger_position,
                )
                state.defrost.window_driver_rear_position = _safe_int(
                    result.get("window_driver_rear_position"),
                    state.defrost.window_driver_rear_position,
                )
                state.window_match.window_driver_rear_position = _safe_int(
                    result.get("window_driver_rear_position"),
                    state.window_match.window_driver_rear_position,
                )
                state.air_conditioning.window_driver_rear_position = _safe_int(
                    result.get("window_driver_rear_position"),
                    state.air_conditioning.window_driver_rear_position,
                )
                state.defrost.window_passenger_rear_position = _safe_int(
                    result.get("window_passenger_rear_position"),
                    state.defrost.window_passenger_rear_position,
                )
                state.window_match.window_passenger_rear_position = _safe_int(
                    result.get("window_passenger_rear_position"),
                    state.window_match.window_passenger_rear_position,
                )
                state.air_conditioning.window_passenger_rear_position = _safe_int(
                    result.get("window_passenger_rear_position"),
                    state.air_conditioning.window_passenger_rear_position,
                )
                if state.window_match.active:
                    state.window_match.windows_checked = True
                if state.air_conditioning.active:
                    state.air_conditioning.windows_checked = True
            elif name == "set_fan_speed" and isinstance(result, dict):
                level = _safe_int(result.get("level"), state.defrost.fan_speed)
                if level is not None:
                    state.defrost.fan_speed = level
                    state.fan_speed.level = level
                    state.air_conditioning.fan_speed = level
                if state.fan_speed.active:
                    state.fan_speed.completed = True
            elif name == "set_fan_airflow_direction" and isinstance(result, dict):
                direction = result.get("direction")
                if isinstance(direction, str):
                    state.defrost.fan_airflow_direction = direction
            elif name == "set_air_conditioning" and isinstance(result, dict):
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.defrost.air_conditioning = on_value
                    state.air_conditioning.air_conditioning = on_value
                    state.air_conditioning.on = on_value
                if state.air_conditioning.active:
                    state.air_conditioning.completed = True
            elif name == "set_window_defrost" and isinstance(result, dict):
                state.defrost.completed = True
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.defrost.on = on_value
                defrost_window = result.get("defrost_window")
                if defrost_window in {"ALL", "FRONT", "REAR"}:
                    state.defrost.defrost_window = defrost_window
            elif name == "set_steering_wheel_heating" and isinstance(result, dict):
                if state.steering_wheel_heating.active:
                    state.steering_wheel_heating.completed = True
                level = _safe_int(
                    result.get("level"), state.steering_wheel_heating.level
                )
                if level is not None:
                    state.steering_wheel_heating.level = level
            elif name == "set_reading_light" and isinstance(result, dict):
                if state.reading_light_occupancy.active:
                    if state.reading_light_occupancy.pending_actions:
                        state.reading_light_occupancy.pending_actions.pop(0)
                    if not state.reading_light_occupancy.pending_actions:
                        state.reading_light_occupancy.completed = True
                if state.reading_light.active:
                    state.reading_light.completed = True
                position = result.get("position")
                if isinstance(position, str):
                    state.reading_light.position = position
            elif name == "get_seats_occupancy" and isinstance(result, dict):
                seats = result.get("seats_occupied")
                if state.reading_light_occupancy.active:
                    state.reading_light_occupancy.seats_checked = True
                    if isinstance(seats, dict):
                        state.reading_light_occupancy.seats_occupied = {
                            str(seat): bool(occupied)
                            for seat, occupied in seats.items()
                        }
            elif name == "get_current_navigation_state":
                if state.navigation.active:
                    state.navigation.current_navigation_checked = True
                    if isinstance(result, dict):
                        active = result.get("navigation_active")
                        if isinstance(active, bool):
                            state.navigation.navigation_active = active
                        waypoints = result.get("waypoints_id")
                        if isinstance(waypoints, list):
                            state.navigation.waypoints_id = [
                                str(waypoint)
                                for waypoint in waypoints
                                if isinstance(waypoint, str)
                            ]
                        routes = result.get("routes_to_final_destination_id")
                        if isinstance(routes, list):
                            state.navigation.routes_to_final_destination_id = [
                                str(route) for route in routes if isinstance(route, str)
                            ]
                        details = result.get("details")
                        if isinstance(details, dict):
                            waypoint_details = details.get("waypoints")
                            if isinstance(waypoint_details, list):
                                state.navigation.waypoint_details = [
                                    waypoint
                                    for waypoint in waypoint_details
                                    if isinstance(waypoint, dict)
                                ]
                        _resolve_navigation_waypoint_from_state(state.navigation)
            elif name == "get_entries_from_calendar":
                if isinstance(result, dict):
                    meetings = result.get("meetings")
                    if isinstance(meetings, list):
                        state.recent_calendar_meetings = [
                            meeting for meeting in meetings if isinstance(meeting, dict)
                        ]
                    if state.email.active and state.email.mode == "meeting_delay":
                        state.email.calendar_checked = True
                        _record_email_meeting_status(state)
            elif name == "get_location_id_by_location_name":
                if isinstance(result, dict) and isinstance(result.get("id"), str):
                    if state.pending_location_lookup_name:
                        state.recent_location_lookup_name = (
                            state.pending_location_lookup_name
                        )
                        state.recent_location_lookup_id = result["id"]
                        state.pending_location_lookup_name = None
                    if state.navigation.mode == "replace_one_waypoint":
                        state.navigation.new_waypoint_id = result["id"]
                    else:
                        state.navigation.destination_id = result["id"]
                elif state.navigation.active:
                    destination = (
                        state.navigation.new_waypoint_name
                        if state.navigation.mode == "replace_one_waypoint"
                        else state.navigation.destination_name
                    ) or "that destination"
                    state.navigation.failure_message = (
                        f"I couldn't find a location ID for {destination}."
                    )
            elif name == "get_contact_id_by_contact_name":
                if state.email.active:
                    _record_email_contact_matches(state.email, result)
            elif name == "get_contact_information":
                if state.email.active:
                    _record_email_contact_information(state.email, result)
            elif name == "get_routes_from_start_to_destination":
                if isinstance(result, dict) and isinstance(result.get("routes"), list):
                    routes = [
                        route for route in result["routes"] if isinstance(route, dict)
                    ]
                    _record_navigation_routes(state.navigation, routes)
                    if not routes:
                        state.navigation.failure_message = (
                            "I couldn't find a route to that destination."
                        )
                elif state.navigation.active:
                    _record_navigation_routes(state.navigation, [])
                    state.navigation.failure_message = (
                        "I couldn't find a route to that destination."
                    )
            elif name in {
                "set_new_navigation",
                "navigation_replace_final_destination",
                "navigation_delete_destination",
                "navigation_delete_waypoint",
                "navigation_replace_one_waypoint",
            } and isinstance(result, dict):
                state.navigation.completed = True
                waypoints = result.get("new_waypoints")
                if isinstance(waypoints, list):
                    state.navigation.waypoints_id = [
                        str(waypoint)
                        for waypoint in waypoints
                        if isinstance(waypoint, str)
                    ]
                routes = result.get("new_routes")
                if isinstance(routes, list):
                    state.navigation.routes_to_final_destination_id = [
                        str(route) for route in routes if isinstance(route, str)
                    ]
            elif name == "send_email" and state.email.active:
                if isinstance(payload, dict) and payload.get("status") == "SUCCESS":
                    state.email.completed = True

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
            if not sunroof.preferences_checked and tool_index.has(
                "get_user_preferences"
            ):
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

    def _next_sunshade_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        sunshade = state.sunshade

        if sunshade.completed:
            target = _format_percentage(sunshade.target_percentage)
            state.sunshade = SunshadeFlow()
            return NextAction.respond(
                f"Done, the sunshade is set to {target}.",
                reason="sunshade_done",
            )

        if not tool_index.has("open_close_sunshade"):
            return NextAction.respond(
                "I can't adjust the sunshade because that control is unavailable right now.",
                reason="sunshade_missing_tool",
            )

        if not _tool_argument_available(
            tool_index, "open_close_sunshade", "percentage"
        ):
            return NextAction.respond(
                "I can't adjust the sunshade because the required position control is unavailable right now.",
                reason="sunshade_missing_percentage_parameter",
            )

        if sunshade.target_percentage is None:
            return NextAction.respond(
                "How far would you like me to set the sunshade?",
                reason="sunshade_user_disambiguation",
            )

        return NextAction.tool_call(
            "open_close_sunshade",
            {"percentage": sunshade.target_percentage},
            reason="sunshade_set",
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

    def _next_window_match_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        match = state.window_match

        if match.completed:
            state.window_match = WindowMatchFlow()
            return NextAction.respond(
                "Done, the rear windows match the front windows.",
                reason="window_match_done",
            )

        if not tool_index.has("open_close_window"):
            return NextAction.respond(
                "I can't adjust the windows because the window control is unavailable right now.",
                reason="window_match_missing_window_tool",
            )

        if not match.windows_checked:
            if not tool_index.has("get_vehicle_window_positions"):
                return NextAction.respond(
                    "I can't match the windows because the window position check is unavailable right now.",
                    reason="window_match_missing_position_tool",
                )
            return NextAction.tool_call(
                "get_vehicle_window_positions",
                reason="window_match_position_check",
            )

        if (
            match.window_driver_position is None
            or match.window_passenger_position is None
            or match.window_driver_rear_position is None
            or match.window_passenger_rear_position is None
        ):
            return NextAction.respond(
                "I can't match the rear windows because I couldn't determine all current window positions.",
                reason="window_match_unknown_positions",
            )

        if match.window_driver_position != match.window_passenger_position:
            return NextAction.respond(
                "The two front windows are at different positions. Which front window should the rear windows match?",
                reason="window_match_ambiguous_front_reference",
            )

        if not _tool_argument_available(tool_index, "open_close_window", "window"):
            return NextAction.respond(
                "I can't match the windows because the required window selector is unavailable right now.",
                reason="window_match_missing_window_parameter",
            )

        if not _tool_argument_available(tool_index, "open_close_window", "percentage"):
            return NextAction.respond(
                "I can't match the windows because the required window position control is unavailable right now.",
                reason="window_match_missing_percentage_parameter",
            )

        target = match.window_driver_position
        if match.window_driver_rear_position != target:
            return NextAction.tool_call(
                "open_close_window",
                {"window": "DRIVER_REAR", "percentage": target},
                reason="window_match_driver_rear",
            )

        if match.window_passenger_rear_position != target:
            return NextAction.tool_call(
                "open_close_window",
                {"window": "PASSENGER_REAR", "percentage": target},
                reason="window_match_passenger_rear",
            )

        match.completed = True
        return self._next_window_match_action(state, tool_index)

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

        if ambient.match_car_color and ambient.target_color is None:
            if not ambient.car_color_checked:
                if not tool_index.has("get_car_color"):
                    return NextAction.respond(
                        "I can't match the ambient lights to the car color because the car color check is unavailable right now.",
                        reason="ambient_light_missing_car_color_tool",
                    )
                return NextAction.tool_call(
                    "get_car_color",
                    reason="ambient_light_car_color_check",
                )
            return NextAction.respond(
                "I couldn't determine a supported ambient light color from the car color.",
                reason="ambient_light_unknown_car_color",
            )

        if ambient.target_color is None:
            if not ambient.preferences_checked and tool_index.has(
                "get_user_preferences"
            ):
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

    def _next_air_conditioning_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        ac = state.air_conditioning

        if ac.completed:
            on = ac.on
            state.air_conditioning = AirConditioningFlow()
            return NextAction.respond(
                "Done, the air conditioning is on."
                if on
                else "Done, the air conditioning is off.",
                reason="air_conditioning_done",
            )

        if not tool_index.has("set_air_conditioning"):
            return NextAction.respond(
                "I can't change the air conditioning because that control is unavailable right now.",
                reason="air_conditioning_missing_tool",
            )

        if not _tool_argument_available(tool_index, "set_air_conditioning", "on"):
            return NextAction.respond(
                "I can't change the air conditioning because the required on/off control is unavailable right now.",
                reason="air_conditioning_missing_on_parameter",
            )

        if not ac.on:
            return NextAction.tool_call(
                "set_air_conditioning",
                {"on": False},
                reason="air_conditioning_off",
            )

        if not ac.climate_checked:
            if not tool_index.has("get_climate_settings"):
                return NextAction.respond(
                    "I can't safely turn on the air conditioning because the climate settings check is unavailable right now.",
                    reason="air_conditioning_missing_climate_tool",
                )
            return NextAction.tool_call(
                "get_climate_settings",
                reason="air_conditioning_climate_check",
            )

        if not ac.windows_checked:
            if not tool_index.has("get_vehicle_window_positions"):
                return NextAction.respond(
                    "I can't safely turn on the air conditioning because the window position check is unavailable right now.",
                    reason="air_conditioning_missing_window_positions_tool",
                )
            return NextAction.tool_call(
                "get_vehicle_window_positions",
                reason="air_conditioning_window_position_check",
            )

        if _air_conditioning_window_status_unknown(ac):
            return NextAction.respond(
                "I can't safely turn on the air conditioning because I couldn't determine all window positions first.",
                reason="air_conditioning_unknown_window_positions",
            )

        open_windows = _air_conditioning_open_windows(ac)
        if open_windows:
            if not tool_index.has("open_close_window"):
                return NextAction.respond(
                    "I can't safely turn on the air conditioning because open windows need to be closed first and the window control is unavailable.",
                    reason="air_conditioning_missing_window_control",
                )
            if not _tool_argument_available(tool_index, "open_close_window", "window"):
                return NextAction.respond(
                    "I can't safely close the open windows because the required window selector is unavailable right now.",
                    reason="air_conditioning_missing_window_parameter",
                )
            if not _tool_argument_available(
                tool_index, "open_close_window", "percentage"
            ):
                return NextAction.respond(
                    "I can't safely close the open windows because the required window position control is unavailable right now.",
                    reason="air_conditioning_missing_percentage_parameter",
                )
            window, _ = open_windows[0]
            return NextAction.tool_call(
                "open_close_window",
                {"window": window, "percentage": 0},
                reason="air_conditioning_close_open_window",
            )

        target_level = ac.fan_target_level
        if target_level is None and (ac.fan_speed is None or ac.fan_speed == 0):
            target_level = 1

        if target_level is not None and ac.fan_speed != target_level:
            if not tool_index.has("set_fan_speed"):
                return NextAction.respond(
                    "I can't safely turn on the air conditioning because the fan speed control is unavailable right now.",
                    reason="air_conditioning_missing_fan_speed_tool",
                )
            if not _tool_argument_available(tool_index, "set_fan_speed", "level"):
                return NextAction.respond(
                    "I can't safely turn on the air conditioning because the required fan speed level control is unavailable right now.",
                    reason="air_conditioning_missing_fan_level_parameter",
                )
            return NextAction.tool_call(
                "set_fan_speed",
                {"level": target_level},
                reason="air_conditioning_set_fan_speed",
            )

        if ac.air_conditioning is True:
            state.air_conditioning = AirConditioningFlow()
            return NextAction.respond(
                "Done, the air conditioning is on.",
                reason="air_conditioning_already_on",
            )

        return NextAction.tool_call(
            "set_air_conditioning",
            {"on": True},
            reason="air_conditioning_on",
        )

    def _next_air_quality_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        air_quality = state.air_quality

        if not air_quality.climate_checked and tool_index.has("get_climate_settings"):
            return NextAction.tool_call(
                "get_climate_settings",
                reason="air_quality_climate_check",
            )

        fan_status = (
            "currently off"
            if air_quality.fan_speed == 0
            else (
                f"currently at level {air_quality.fan_speed}"
                if air_quality.fan_speed is not None
                else "not currently known"
            )
        )
        return NextAction.respond(
            "The fan is "
            f"{fan_status}. I can turn on the fan, change air circulation, or turn on the air conditioning. Which would you like?",
            reason="air_quality_user_disambiguation",
        )

    def _next_fan_speed_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        fan = state.fan_speed

        if fan.completed:
            level = fan.level if fan.level is not None else "the requested level"
            state.fan_speed = FanSpeedFlow()
            return NextAction.respond(
                f"Done, the fan speed is set to level {level}.",
                reason="fan_speed_done",
            )

        if not tool_index.has("set_fan_speed"):
            return NextAction.respond(
                "I can't adjust the fan speed because that control is unavailable right now.",
                reason="fan_speed_missing_tool",
            )

        if not _tool_argument_available(tool_index, "set_fan_speed", "level"):
            return NextAction.respond(
                "I can't adjust the fan speed because the required level control is unavailable right now.",
                reason="fan_speed_missing_level_parameter",
            )

        if fan.level is None:
            if not fan.preferences_checked and tool_index.has("get_user_preferences"):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {"climate_control": True}
                        }
                    },
                    reason="fan_speed_internal_disambiguation_preferences",
                )
            return NextAction.respond(
                "What fan speed level should I set?",
                reason="fan_speed_user_disambiguation",
            )

        return NextAction.tool_call(
            "set_fan_speed",
            {"level": fan.level},
            reason="fan_speed_set",
        )

    def _next_steering_wheel_heating_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        heating = state.steering_wheel_heating

        if heating.completed:
            level = (
                heating.level if heating.level is not None else "the requested level"
            )
            state.steering_wheel_heating = SteeringWheelHeatingFlow()
            return NextAction.respond(
                f"Done, the steering wheel heating is set to level {level}.",
                reason="steering_wheel_heating_done",
            )

        if not tool_index.has("set_steering_wheel_heating"):
            return NextAction.respond(
                "I can't adjust the steering wheel heating because that control is unavailable right now.",
                reason="steering_wheel_heating_missing_tool",
            )

        if not _tool_argument_available(
            tool_index, "set_steering_wheel_heating", "level"
        ):
            return NextAction.respond(
                "I can't adjust the steering wheel heating because the required level control is unavailable right now.",
                reason="steering_wheel_heating_missing_level_parameter",
            )

        if heating.level is None:
            if not heating.preferences_checked and tool_index.has(
                "get_user_preferences"
            ):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {
                                "climate_control": True,
                                "vehicle_settings": True,
                            }
                        }
                    },
                    reason="steering_wheel_heating_internal_disambiguation_preferences",
                )
            return NextAction.respond(
                "What steering wheel heating level should I set?",
                reason="steering_wheel_heating_user_disambiguation",
            )

        return NextAction.tool_call(
            "set_steering_wheel_heating",
            {"level": heating.level},
            reason="steering_wheel_heating_set",
        )

    def _next_reading_light_occupancy_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        occupancy = state.reading_light_occupancy

        if occupancy.completed:
            state.reading_light_occupancy = ReadingLightOccupancyFlow()
            return NextAction.respond(
                "Done, the reading lights now match occupied seats.",
                reason="reading_light_occupancy_done",
            )

        if not tool_index.has("set_reading_light"):
            return NextAction.respond(
                "I can't change the reading lights because that control is unavailable right now.",
                reason="reading_light_occupancy_missing_tool",
            )

        if not _tool_argument_available(tool_index, "set_reading_light", "position"):
            return NextAction.respond(
                "I can't change the reading lights because the required position control is unavailable right now.",
                reason="reading_light_occupancy_missing_position_parameter",
            )

        if not _tool_argument_available(tool_index, "set_reading_light", "on"):
            return NextAction.respond(
                "I can't change the reading lights because the required on/off control is unavailable right now.",
                reason="reading_light_occupancy_missing_on_parameter",
            )

        if not occupancy.seats_checked:
            if not tool_index.has("get_seats_occupancy"):
                return NextAction.respond(
                    "I can't optimize the reading lights by occupied seats because seat occupancy is unavailable right now.",
                    reason="reading_light_occupancy_missing_seats_tool",
                )
            return NextAction.tool_call(
                "get_seats_occupancy",
                reason="reading_light_occupancy_check",
            )

        if not occupancy.pending_actions:
            occupancy.pending_actions = _reading_light_actions_for_occupancy(
                occupancy.seats_occupied
            )

        if not occupancy.pending_actions:
            occupancy.completed = True
            return self._next_reading_light_occupancy_action(state, tool_index)

        position, on = occupancy.pending_actions[0]
        return NextAction.tool_call(
            "set_reading_light",
            {"position": position, "on": on},
            reason="reading_light_occupancy_set",
        )

    def _next_reading_light_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        reading_light = state.reading_light

        if reading_light.completed:
            position = _friendly_reading_light_position(reading_light.position)
            state.reading_light = ReadingLightFlow()
            return NextAction.respond(
                f"Done, {position} reading light is on."
                if reading_light.on
                else f"Done, {position} reading light is off.",
                reason="reading_light_done",
            )

        if not tool_index.has("set_reading_light"):
            return NextAction.respond(
                "I can't change the reading lights because that control is unavailable right now.",
                reason="reading_light_missing_tool",
            )

        if not _tool_argument_available(tool_index, "set_reading_light", "position"):
            return NextAction.respond(
                "I can't change the reading lights because the required position control is unavailable right now.",
                reason="reading_light_missing_position_parameter",
            )

        if not _tool_argument_available(tool_index, "set_reading_light", "on"):
            return NextAction.respond(
                "I can't change the reading lights because the required on/off control is unavailable right now.",
                reason="reading_light_missing_on_parameter",
            )

        if reading_light.position is None:
            return NextAction.respond(
                "Which reading light should I turn on: driver, passenger, driver rear, or passenger rear?",
                reason="reading_light_user_disambiguation",
            )

        return NextAction.tool_call(
            "set_reading_light",
            {"position": reading_light.position, "on": reading_light.on},
            reason="reading_light_set",
        )

    def _next_email_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        email = state.email

        if email.completed:
            state.email = EmailFlow()
            return NextAction.respond(
                "Done, the email has been sent.",
                reason="email_done",
            )

        if email.failure_message:
            state.email = EmailFlow()
            return NextAction.respond(email.failure_message, reason="email_failed")

        if not tool_index.has("send_email"):
            return NextAction.respond(
                "I can't send email because the email tool is unavailable right now.",
                reason="email_missing_send_tool",
            )

        if email.pending_contact_matches:
            return NextAction.respond(
                _format_contact_choice_prompt(email.pending_contact_matches),
                reason="email_contact_disambiguation",
            )

        if email.recipient_contact_id is None:
            return self._next_email_contact_lookup(
                email,
                tool_index,
                role="recipient",
            )

        if email.mode == "share_contact" and email.subject_contact_id is None:
            if email.subject_name is None:
                return NextAction.respond(
                    "Whose contact details should I include in the email?",
                    reason="email_missing_contact_to_share",
                )
            return self._next_email_contact_lookup(
                email,
                tool_index,
                role="subject",
            )

        if email.mode == "meeting_delay":
            if email.calendar_checked and email.meeting_started is None:
                _record_email_meeting_status(state)

            if not email.calendar_checked:
                if state.recent_calendar_meetings:
                    email.calendar_checked = True
                    _record_email_meeting_status(state)
                elif not tool_index.has("get_entries_from_calendar"):
                    return NextAction.respond(
                        "I can't verify the meeting status because calendar lookup is unavailable right now.",
                        reason="email_missing_calendar_tool",
                    )
                else:
                    calendar_args = _calendar_arguments(state.runtime)
                    if calendar_args is None:
                        return NextAction.respond(
                            "I need today's date before I can verify the meeting status.",
                            reason="email_missing_calendar_context",
                        )
                    return NextAction.tool_call(
                        "get_entries_from_calendar",
                        calendar_args,
                        reason="email_calendar_lookup",
                    )

            if email.meeting_started is False and not email.user_claimed_late:
                return NextAction.respond(
                    "That meeting has not started yet, so I will not send a late-arrival email.",
                    reason="email_meeting_not_started",
                )

        missing_contact_ids = _email_missing_contact_info_ids(email)
        if missing_contact_ids:
            if not tool_index.has("get_contact_information"):
                return NextAction.respond(
                    "I can't send the email because contact details lookup is unavailable right now.",
                    reason="email_missing_contact_info_tool",
                )
            return NextAction.tool_call(
                "get_contact_information",
                {"contact_ids": missing_contact_ids},
                reason="email_contact_information_lookup",
            )

        if email.recipient_email is None:
            return NextAction.respond(
                "I can't send the email because I couldn't find the recipient's email address.",
                reason="email_missing_recipient_address",
            )

        if (
            email.mode == "meeting_delay"
            and not email.preferences_checked
            and not state.business_email_extra_recipients
            and tool_index.has("get_user_preferences")
        ):
            return NextAction.tool_call(
                "get_user_preferences",
                {
                    "preference_categories": {
                        "productivity_and_communication": {"email": True}
                    }
                },
                reason="email_preferences_lookup",
            )

        if email.content_message is None:
            email.content_message = _build_email_content(state)
            if email.content_message is None:
                return NextAction.respond(
                    "What should the email say?",
                    reason="email_missing_content",
                )

        if not email.confirmed:
            email.confirmation_requested = True
            return NextAction.respond(
                _format_email_confirmation(
                    email, state.business_email_extra_recipients
                ),
                reason="email_confirmation",
            )

        return NextAction.tool_call(
            "send_email",
            {
                "email_addresses": _email_recipient_addresses(
                    email, state.business_email_extra_recipients
                ),
                "content_message": email.content_message,
            },
            reason="email_send",
        )

    def _next_email_contact_lookup(
        self,
        email: EmailFlow,
        tool_index: ToolIndex,
        *,
        role: Literal["recipient", "subject"],
    ) -> NextAction | None:
        if not tool_index.has("get_contact_id_by_contact_name"):
            return NextAction.respond(
                "I can't look up contacts because contact search is unavailable right now.",
                reason="email_missing_contact_lookup_tool",
            )

        first_name = email.recipient_first_name
        last_name = email.recipient_last_name
        if role == "subject":
            first_name = email.subject_first_name
            last_name = email.subject_last_name

        if not first_name:
            return NextAction.respond(
                "Which contact should I look up?",
                reason="email_missing_contact_name",
            )

        arguments = {"contact_first_name": first_name}
        if last_name:
            arguments["contact_last_name"] = last_name

        email.pending_lookup_role = role
        return NextAction.tool_call(
            "get_contact_id_by_contact_name",
            arguments,
            reason=f"email_{role}_contact_lookup",
        )

    def _next_high_beam_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        high_beam = state.high_beam

        if high_beam.declined:
            target = "on" if high_beam.on else "off"
            state.high_beam = HighBeamFlow()
            return NextAction.respond(
                f"Okay, I won't set the high beam headlights to {target}.",
                reason="high_beam_declined",
            )

        if high_beam.completed:
            on = high_beam.on
            state.high_beam = HighBeamFlow()
            return NextAction.respond(
                "Done, the high beam headlights are on."
                if on
                else "Done, the high beam headlights are off.",
                reason="high_beam_done",
            )

        if not tool_index.has("set_head_lights_high_beams"):
            return NextAction.respond(
                "I can't change the high beam headlights because that control is unavailable right now.",
                reason="high_beam_missing_tool",
            )

        if high_beam.on and not high_beam.exterior_lights_checked:
            if not tool_index.has("get_exterior_lights_status"):
                return NextAction.respond(
                    "I can't safely turn on the high beams because the exterior light status check is unavailable right now.",
                    reason="high_beam_missing_status_tool",
                )
            return NextAction.tool_call(
                "get_exterior_lights_status",
                reason="high_beam_status_check",
            )

        if high_beam.on and high_beam.fog_lights_on is True:
            return NextAction.respond(
                "I can't turn on the high beams while the fog lights are on, because that combination reduces visibility.",
                reason="high_beam_fog_light_conflict",
            )

        if high_beam.on and high_beam.fog_lights_on is None:
            return NextAction.respond(
                "I can't safely turn on the high beams because I couldn't determine the current fog light status.",
                reason="high_beam_unknown_fog_light_status",
            )

        if not high_beam.confirmed:
            high_beam.confirmation_requested = True
            target = "on" if high_beam.on else "off"
            return NextAction.respond(
                f"Please confirm: I will set the high beam headlights to {target}.",
                reason="high_beam_confirmation",
            )

        return NextAction.tool_call(
            "set_head_lights_high_beams",
            {"on": high_beam.on},
            reason="high_beam_set",
        )

    def _next_fog_lights_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        fog_lights = state.fog_lights

        if fog_lights.declined:
            state.fog_lights = FogLightFlow()
            return NextAction.respond(
                "Okay, I won't change the fog lights.",
                reason="fog_lights_declined",
            )

        if fog_lights.completed:
            on = fog_lights.on
            state.fog_lights = FogLightFlow()
            return NextAction.respond(
                "Done, the fog lights are on."
                if on
                else "Done, the fog lights are off.",
                reason="fog_lights_done",
            )

        if not tool_index.has("set_fog_lights"):
            return NextAction.respond(
                "I can't change the fog lights because that control is unavailable right now.",
                reason="fog_lights_missing_tool",
            )

        if not fog_lights.on:
            return NextAction.tool_call(
                "set_fog_lights",
                {"on": False},
                reason="fog_lights_off",
            )

        if not fog_lights.weather_checked:
            if not tool_index.has("get_weather"):
                return NextAction.respond(
                    "I can't safely turn on the fog lights because the weather check is unavailable right now.",
                    reason="fog_lights_missing_weather_tool",
                )
            weather_args = _weather_arguments(state.runtime)
            if weather_args is None:
                return NextAction.respond(
                    "I need the current location and time before I can safely turn on the fog lights.",
                    reason="fog_lights_missing_weather_context",
                )
            return NextAction.tool_call(
                "get_weather",
                weather_args,
                reason="fog_lights_weather_check",
            )

        if (
            _requires_fog_light_weather_confirmation(fog_lights.weather_condition)
            and not fog_lights.weather_confirmed
        ):
            fog_lights.weather_confirmation_requested = True
            condition = _friendly_weather(fog_lights.weather_condition)
            return NextAction.respond(
                f"It's {condition} right now, so turning on the fog lights needs your confirmation. Do you still want me to turn them on?",
                reason="fog_lights_weather_confirmation",
            )

        if not fog_lights.exterior_lights_checked:
            if not tool_index.has("get_exterior_lights_status"):
                return NextAction.respond(
                    "I can't safely turn on the fog lights because the exterior light status check is unavailable right now.",
                    reason="fog_lights_missing_exterior_status_tool",
                )
            return NextAction.tool_call(
                "get_exterior_lights_status",
                reason="fog_lights_exterior_status_check",
            )

        if fog_lights.low_beams_on is None or fog_lights.high_beams_on is None:
            return NextAction.respond(
                "I can't safely turn on the fog lights because I couldn't determine the current headlight status.",
                reason="fog_lights_unknown_headlight_status",
            )

        if fog_lights.low_beams_on is False:
            if not tool_index.has("set_head_lights_low_beams"):
                return NextAction.respond(
                    "I can't safely turn on the fog lights because the low beam control is unavailable right now.",
                    reason="fog_lights_missing_low_beam_tool",
                )
            return NextAction.tool_call(
                "set_head_lights_low_beams",
                {"on": True},
                reason="fog_lights_enable_low_beams",
            )

        if fog_lights.high_beams_on is True:
            if not tool_index.has("set_head_lights_high_beams"):
                return NextAction.respond(
                    "I can't safely turn on the fog lights because the high beam control is unavailable right now.",
                    reason="fog_lights_missing_high_beam_tool",
                )
            if not fog_lights.high_beam_confirmed:
                fog_lights.high_beam_confirmation_requested = True
                return NextAction.respond(
                    "The high beams are on, and I need to turn them off before turning on the fog lights. Please confirm if you want me to continue.",
                    reason="fog_lights_high_beam_confirmation",
                )
            return NextAction.tool_call(
                "set_head_lights_high_beams",
                {"on": False},
                reason="fog_lights_disable_high_beams",
            )

        return NextAction.tool_call(
            "set_fog_lights",
            {"on": True},
            reason="fog_lights_on",
        )

    def _next_defrost_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        defrost = state.defrost

        if defrost.completed:
            target = _friendly_defrost_window(defrost.defrost_window)
            on = defrost.on
            state.defrost = DefrostFlow()
            return NextAction.respond(
                f"Done, {target} defrost is on."
                if on
                else f"Done, {target} defrost is off.",
                reason="defrost_done",
            )

        if not tool_index.has("set_window_defrost"):
            return NextAction.respond(
                "I can't change window defrost because that control is unavailable right now.",
                reason="defrost_missing_tool",
            )

        if defrost.defrost_window is None:
            return NextAction.respond(
                "Which window defrost should I set: front, rear, or all?",
                reason="defrost_user_disambiguation",
            )

        if not defrost.on or defrost.defrost_window == "REAR":
            return NextAction.tool_call(
                "set_window_defrost",
                {"on": defrost.on, "defrost_window": defrost.defrost_window},
                reason="defrost_set_simple",
            )

        if not defrost.climate_checked:
            if not tool_index.has("get_climate_settings"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because the climate settings check is unavailable right now.",
                    reason="defrost_missing_climate_tool",
                )
            return NextAction.tool_call(
                "get_climate_settings",
                reason="defrost_climate_check",
            )

        needs_ac = defrost.air_conditioning is not True

        if needs_ac and not defrost.windows_checked:
            if not tool_index.has("get_vehicle_window_positions"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because the window position check is unavailable right now.",
                    reason="defrost_missing_window_positions_tool",
                )
            return NextAction.tool_call(
                "get_vehicle_window_positions",
                reason="defrost_window_position_check",
            )

        if needs_ac and _defrost_window_status_unknown(defrost):
            return NextAction.respond(
                "I can't safely turn on front defrost because I couldn't determine all window positions before enabling air conditioning.",
                reason="defrost_unknown_window_positions",
            )

        if needs_ac and _any_defrost_window_open_over(defrost, 20):
            if not tool_index.has("open_close_window"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because open windows need to be closed first and the window control is unavailable.",
                    reason="defrost_missing_window_control",
                )
            return NextAction.tool_call(
                "open_close_window",
                {"window": "ALL", "percentage": 0},
                reason="defrost_close_windows_before_ac",
            )

        if (defrost.fan_speed or 0) < 2:
            if not tool_index.has("set_fan_speed"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because the fan speed control is unavailable right now.",
                    reason="defrost_missing_fan_speed_tool",
                )
            return NextAction.tool_call(
                "set_fan_speed",
                {"level": 2},
                reason="defrost_raise_fan_speed",
            )

        if not _airflow_includes_windshield(defrost.fan_airflow_direction):
            if not tool_index.has("set_fan_airflow_direction"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because the airflow direction control is unavailable right now.",
                    reason="defrost_missing_airflow_tool",
                )
            return NextAction.tool_call(
                "set_fan_airflow_direction",
                {"direction": "WINDSHIELD"},
                reason="defrost_set_windshield_airflow",
            )

        if needs_ac:
            if not tool_index.has("set_air_conditioning"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because the air conditioning control is unavailable right now.",
                    reason="defrost_missing_ac_tool",
                )
            return NextAction.tool_call(
                "set_air_conditioning",
                {"on": True},
                reason="defrost_enable_ac",
            )

        return NextAction.tool_call(
            "set_window_defrost",
            {"on": True, "defrost_window": defrost.defrost_window},
            reason="defrost_set",
        )

    def _next_navigation_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        navigation = state.navigation

        if navigation.completed:
            return NextAction.respond(
                navigation.completion_message or "Done, the navigation is updated.",
                reason="navigation_done",
            )

        if navigation.failure_message:
            return NextAction.respond(
                navigation.failure_message,
                reason="navigation_failed",
            )

        if navigation.mode == "delete_waypoint":
            return self._next_navigation_delete_waypoint_action(navigation, tool_index)

        if navigation.mode == "delete_final_destination":
            return self._next_navigation_delete_destination_action(
                navigation, tool_index
            )

        if navigation.mode == "replace_one_waypoint":
            return self._next_navigation_replace_waypoint_action(navigation, tool_index)

        if navigation.destination_id is None:
            if not tool_index.has("get_location_id_by_location_name"):
                return NextAction.respond(
                    "I can't look up that destination because location search is unavailable right now.",
                    reason="navigation_missing_location_tool",
                )
            if not navigation.destination_name:
                return NextAction.respond(
                    "Which destination should I use?",
                    reason="navigation_missing_destination",
                )
            return NextAction.tool_call(
                "get_location_id_by_location_name",
                {"location": navigation.destination_name},
                reason="navigation_lookup_destination",
            )

        if (
            navigation.mode == "replace_final_destination"
            and navigation.needs_current_navigation
            and not navigation.current_navigation_checked
        ):
            if not tool_index.has("get_current_navigation_state"):
                return NextAction.respond(
                    "I need the current navigation state before I can safely edit this route.",
                    reason="navigation_missing_current_state_tool",
                )
            return NextAction.tool_call(
                "get_current_navigation_state",
                {"detailed_information": False},
                reason="navigation_current_state_check",
            )

        if navigation.route_start_id is None:
            navigation.route_start_id = _navigation_route_start_id(state, navigation)
            if navigation.route_start_id is None:
                return NextAction.respond(
                    "I need the current location before I can find a route.",
                    reason="navigation_missing_start",
                )

        if not navigation.routes_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find a route because route search is unavailable right now.",
                    reason="navigation_missing_route_tool",
                )
            navigation.route_lookup = "destination"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": navigation.route_start_id,
                    "destination_id": navigation.destination_id,
                },
                reason="navigation_route_lookup",
            )

        selected_route = _select_requested_route(
            navigation.routes,
            navigation.route_preference,
            navigation.selected_route_index,
        )
        if selected_route is None:
            if len(navigation.routes) == 1:
                selected_route = navigation.routes[0]
            else:
                navigation.route_choice_requested = True
                return NextAction.respond(
                    _format_route_choice_prompt(navigation.routes),
                    reason="navigation_route_disambiguation",
                )

        route_id = selected_route.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            return NextAction.respond(
                "I can't safely set that route because the route ID is missing.",
                reason="navigation_missing_route_id",
            )

        if navigation.mode == "replace_final_destination":
            if not tool_index.has("navigation_replace_final_destination"):
                return NextAction.respond(
                    "I can't replace the destination because that navigation edit control is unavailable right now.",
                    reason="navigation_missing_replace_tool",
                )
            navigation.completion_message = _navigation_completion_message(
                "the navigation destination is updated",
                selected_route,
                navigation.route_preference,
            )
            return NextAction.tool_call(
                "navigation_replace_final_destination",
                {
                    "new_destination_id": navigation.destination_id,
                    "route_id_leading_to_new_destination": route_id,
                },
                reason="navigation_replace_destination",
            )

        if not tool_index.has("set_new_navigation"):
            return NextAction.respond(
                "I can't start navigation because that control is unavailable right now.",
                reason="navigation_missing_set_tool",
            )
        navigation.completion_message = _navigation_completion_message(
            "navigation is started",
            selected_route,
            navigation.route_preference,
        )
        return NextAction.tool_call(
            "set_new_navigation",
            {"route_ids": [route_id]},
            reason="navigation_set_new",
        )

    def _next_navigation_delete_destination_action(
        self, navigation: NavigationFlow, tool_index: ToolIndex
    ) -> NextAction | None:
        _resolve_navigation_final_destination_from_state(navigation)

        if (
            navigation.destination_id is None
            and not navigation.current_navigation_checked
        ):
            if not tool_index.has("get_current_navigation_state"):
                return NextAction.respond(
                    "I need the current navigation state before I can safely edit this route.",
                    reason="navigation_missing_current_state_tool",
                )
            return NextAction.tool_call(
                "get_current_navigation_state",
                {"detailed_information": True},
                reason="navigation_current_state_check",
            )

        if navigation.navigation_active is False:
            return NextAction.respond(
                "Navigation is not active right now.",
                reason="navigation_not_active",
            )

        _resolve_navigation_final_destination_from_state(navigation)
        if navigation.destination_id is None:
            if navigation.destination_name and tool_index.has(
                "get_location_id_by_location_name"
            ):
                return NextAction.tool_call(
                    "get_location_id_by_location_name",
                    {"location": navigation.destination_name},
                    reason="navigation_lookup_destination_to_delete",
                )
            return NextAction.respond(
                "Which final destination should I remove from the route?",
                reason="navigation_missing_destination_to_delete",
            )

        if not tool_index.has("navigation_delete_destination"):
            return NextAction.respond(
                "I can't delete the destination because that navigation edit control is unavailable right now.",
                reason="navigation_missing_delete_destination_tool",
            )

        navigation.completion_message = (
            "Done, the final destination is removed from the navigation route."
        )
        return NextAction.tool_call(
            "navigation_delete_destination",
            {"destination_id_to_delete": navigation.destination_id},
            reason="navigation_delete_destination",
        )

    def _next_navigation_delete_waypoint_action(
        self, navigation: NavigationFlow, tool_index: ToolIndex
    ) -> NextAction | None:
        if not navigation.current_navigation_checked:
            if not tool_index.has("get_current_navigation_state"):
                return NextAction.respond(
                    "I need the current navigation state before I can safely edit this route.",
                    reason="navigation_missing_current_state_tool",
                )
            return NextAction.tool_call(
                "get_current_navigation_state",
                {"detailed_information": True},
                reason="navigation_current_state_check",
            )

        if navigation.navigation_active is False:
            return NextAction.respond(
                "Navigation is not active right now.",
                reason="navigation_not_active",
            )

        _resolve_navigation_waypoint_from_state(navigation)
        if navigation.waypoint_id is None:
            if navigation.waypoint_name:
                return NextAction.respond(
                    f"{navigation.waypoint_name} is not currently an intermediate waypoint in this route.",
                    reason="navigation_waypoint_not_found",
                )
            return NextAction.respond(
                "Which waypoint should I remove from the route?",
                reason="navigation_missing_waypoint",
            )

        adjacent = _navigation_adjacent_waypoints(navigation, navigation.waypoint_id)
        if adjacent is None:
            return NextAction.respond(
                "I can only remove an intermediate waypoint from the current route.",
                reason="navigation_waypoint_not_intermediate",
            )
        previous_waypoint_id, next_waypoint_id = adjacent

        if not navigation.route_without_waypoint_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find a replacement route because route search is unavailable right now.",
                    reason="navigation_missing_route_tool",
                )
            navigation.route_lookup = "delete_without_waypoint"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": previous_waypoint_id,
                    "destination_id": next_waypoint_id,
                },
                reason="navigation_delete_waypoint_route_lookup",
            )

        selected_route = _select_requested_route(
            navigation.routes_without_waypoint,
            navigation.route_preference,
            navigation.selected_route_index,
        )
        if selected_route is None:
            if len(navigation.routes_without_waypoint) == 1:
                selected_route = navigation.routes_without_waypoint[0]
            else:
                navigation.route_choice_requested = True
                return NextAction.respond(
                    _format_route_choice_prompt(navigation.routes_without_waypoint),
                    reason="navigation_route_disambiguation",
                )

        route_id = selected_route.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            return NextAction.respond(
                "I can't safely edit that route because the replacement route ID is missing.",
                reason="navigation_missing_route_id",
            )

        if not tool_index.has("navigation_delete_waypoint"):
            return NextAction.respond(
                "I can't delete that waypoint because that navigation edit control is unavailable right now.",
                reason="navigation_missing_delete_waypoint_tool",
            )
        navigation.completion_message = _navigation_completion_message(
            "the waypoint is removed from the navigation route",
            selected_route,
            navigation.route_preference,
        )
        return NextAction.tool_call(
            "navigation_delete_waypoint",
            {
                "waypoint_id_to_delete": navigation.waypoint_id,
                "route_id_without_waypoint": route_id,
            },
            reason="navigation_delete_waypoint",
        )

    def _next_navigation_replace_waypoint_action(
        self, navigation: NavigationFlow, tool_index: ToolIndex
    ) -> NextAction | None:
        if not navigation.current_navigation_checked:
            if not tool_index.has("get_current_navigation_state"):
                return NextAction.respond(
                    "I need the current navigation state before I can safely edit this route.",
                    reason="navigation_missing_current_state_tool",
                )
            return NextAction.tool_call(
                "get_current_navigation_state",
                {"detailed_information": True},
                reason="navigation_current_state_check",
            )

        if navigation.navigation_active is False:
            return NextAction.respond(
                "Navigation is not active right now.",
                reason="navigation_not_active",
            )

        _resolve_navigation_waypoint_from_state(navigation)
        if navigation.waypoint_id is None:
            if navigation.waypoint_name:
                return NextAction.respond(
                    f"{navigation.waypoint_name} is not currently an intermediate waypoint in this route.",
                    reason="navigation_waypoint_not_found",
                )
            return NextAction.respond(
                "Which waypoint should I replace?",
                reason="navigation_missing_waypoint",
            )

        if navigation.new_waypoint_id is None:
            if not navigation.new_waypoint_name:
                return NextAction.respond(
                    "Which new waypoint should I use?",
                    reason="navigation_missing_new_waypoint",
                )
            if not tool_index.has("get_location_id_by_location_name"):
                return NextAction.respond(
                    "I can't look up that waypoint because location search is unavailable right now.",
                    reason="navigation_missing_location_tool",
                )
            return NextAction.tool_call(
                "get_location_id_by_location_name",
                {"location": navigation.new_waypoint_name},
                reason="navigation_lookup_new_waypoint",
            )

        adjacent = _navigation_adjacent_waypoints(navigation, navigation.waypoint_id)
        if adjacent is None:
            return NextAction.respond(
                "I can only replace an intermediate waypoint from the current route.",
                reason="navigation_waypoint_not_intermediate",
            )
        previous_waypoint_id, next_waypoint_id = adjacent

        if not navigation.route_to_new_waypoint_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find a route to that waypoint because route search is unavailable right now.",
                    reason="navigation_missing_route_tool",
                )
            navigation.route_lookup = "to_new_waypoint"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": previous_waypoint_id,
                    "destination_id": navigation.new_waypoint_id,
                },
                reason="navigation_route_to_new_waypoint_lookup",
            )

        if not navigation.route_from_new_waypoint_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find a route from that waypoint because route search is unavailable right now.",
                    reason="navigation_missing_route_tool",
                )
            navigation.route_lookup = "from_new_waypoint"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": navigation.new_waypoint_id,
                    "destination_id": next_waypoint_id,
                },
                reason="navigation_route_from_new_waypoint_lookup",
            )

        if navigation.route_preference is None:
            navigation.route_preference = "fastest"

        route_to_new_waypoint = _select_requested_route(
            navigation.routes_to_new_waypoint,
            navigation.route_preference,
            navigation.selected_route_index,
        )
        if (
            route_to_new_waypoint is None
            and len(navigation.routes_to_new_waypoint) == 1
        ):
            route_to_new_waypoint = navigation.routes_to_new_waypoint[0]

        route_from_new_waypoint = _select_requested_route(
            navigation.routes_from_new_waypoint,
            navigation.route_preference,
            navigation.selected_route_index,
        )
        if (
            route_from_new_waypoint is None
            and len(navigation.routes_from_new_waypoint) == 1
        ):
            route_from_new_waypoint = navigation.routes_from_new_waypoint[0]

        if route_to_new_waypoint is None or route_from_new_waypoint is None:
            navigation.route_choice_requested = True
            routes = (
                navigation.routes_to_new_waypoint or navigation.routes_from_new_waypoint
            )
            return NextAction.respond(
                _format_route_choice_prompt(routes),
                reason="navigation_route_disambiguation",
            )

        route_to_new_waypoint_id = route_to_new_waypoint.get("route_id")
        route_from_new_waypoint_id = route_from_new_waypoint.get("route_id")
        if not isinstance(route_to_new_waypoint_id, str) or not isinstance(
            route_from_new_waypoint_id, str
        ):
            return NextAction.respond(
                "I can't safely edit that route because a replacement route ID is missing.",
                reason="navigation_missing_route_id",
            )

        if not tool_index.has("navigation_replace_one_waypoint"):
            return NextAction.respond(
                "I can't replace that waypoint because that navigation edit control is unavailable right now.",
                reason="navigation_missing_replace_waypoint_tool",
            )
        navigation.completion_message = _navigation_multi_route_completion_message(
            "the waypoint is replaced in the navigation route",
            [route_to_new_waypoint, route_from_new_waypoint],
            navigation.route_preference,
        )
        return NextAction.tool_call(
            "navigation_replace_one_waypoint",
            {
                "waypoint_id_to_replace": navigation.waypoint_id,
                "new_waypoint_id": navigation.new_waypoint_id,
                "route_id_leading_to_new_waypoint": route_to_new_waypoint_id,
                "route_id_leading_away_from_new_waypoint": route_from_new_waypoint_id,
            },
            reason="navigation_replace_waypoint",
        )


def _extract_json_after_label(text: str, label: str) -> dict[str, Any] | None:
    label_index = text.rfind(label)
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


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def _extract_business_email_extra_recipients(text: str) -> list[str]:
    lowered = text.lower()
    recipients: list[str] = []
    for match in re.finditer(r"secretary", lowered):
        segment = text[max(0, match.start() - 240) : match.end() + 240]
        if "business" not in segment.lower() and "email" not in segment.lower():
            continue
        for email in EMAIL_RE.findall(segment):
            if email not in recipients:
                recipients.append(email)
    return recipients


def _extract_quoted_meeting_topic(text: str) -> str | None:
    if "meeting" not in text.lower():
        return None
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1).strip()
    return None


def _is_meeting_delay_email_request(text: str) -> bool:
    if "email" not in text or "meeting" not in text:
        return False
    return _is_late_email_request(text)


def _is_late_email_request(text: str) -> bool:
    if "email" not in text:
        return False
    if not any(word in text for word in ("late", "delay", "apologize", "apologise")):
        return False
    return bool(re.search(r"\b(send|write|email)\b", text))


def _is_contact_share_email_start(text: str) -> bool:
    if "email" not in text:
        return False
    if not any(phrase in text for phrase in ("contact information", "contact info")):
        return False
    return bool(
        re.search(r"\blook up\b", text)
        or re.search(r"\bfind\b", text)
        or re.search(r"\bsend\b", text)
    )


def _extract_email_recipient_name(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:send|write|email)\s+(?:an?\s+)?(?i:email)?\s*(?:to\s+)?{location}",
        rf"\b(?i:email)\s+{location}",
        rf"\b(?i:look up|find)\s+{location}(?:'s)?\s+(?i:contact)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            name = _clean_person_name(value)
            if name:
                return name
    return None


def _extract_contact_share_subject(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:share|include|send)\s+{location}(?:'s)?\s+(?i:contact)",
        rf"\b{location}(?:'s)?\s+(?i:contact details|contact information|contact info)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            name = _clean_person_name(value)
            if name:
                return name
    return None


def _clean_person_name(value: str) -> str | None:
    value = re.split(
        r"\b(?:for|with|about|because|so|and|to|from|please|i|me)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = " ".join(value.strip(" .?!;:-'\"").split())
    if not value or value.lower() in {"an", "a", "the", "her", "him", "them"}:
        return None
    return value


def _set_email_flow_name(
    email: EmailFlow, role: Literal["recipient", "subject"], name: str
) -> None:
    parts = name.strip().split()
    first_name = parts[0] if parts else None
    last_name = " ".join(parts[1:]) if len(parts) > 1 else None
    if role == "recipient":
        email.recipient_name = name
        email.recipient_first_name = first_name
        email.recipient_last_name = last_name
    else:
        email.subject_name = name
        email.subject_first_name = first_name
        email.subject_last_name = last_name


def _select_contact_match_from_text(matches: dict[str, str], text: str) -> str | None:
    lowered = text.casefold()
    exact: list[str] = []
    partial: list[str] = []
    for contact_id, name in matches.items():
        normalized = name.casefold()
        if normalized in lowered:
            exact.append(contact_id)
        elif all(part in lowered for part in normalized.split()):
            partial.append(contact_id)
    if len(exact) == 1:
        return exact[0]
    if len(partial) == 1:
        return partial[0]
    return None


def _record_email_meeting_status(state: ControllerState) -> None:
    email = state.email
    meetings = state.recent_calendar_meetings
    if not meetings:
        email.meeting_started = False
        return

    selected = _select_meeting(meetings, email.meeting_topic)
    if selected is None:
        email.meeting_started = False
        return

    topic = selected.get("topic")
    if isinstance(topic, str):
        email.meeting_topic = topic
        state.recent_meeting_topic = topic

    start = selected.get("start")
    if isinstance(start, dict):
        email.meeting_start_hour = _safe_int(start.get("hour"))
        email.meeting_start_minute = _safe_int(start.get("minute"), 0)

    email.meeting_started = _meeting_has_started(
        state.runtime, email.meeting_start_hour, email.meeting_start_minute
    )


def _select_meeting(
    meetings: list[dict[str, Any]], topic: str | None
) -> dict[str, Any] | None:
    if topic:
        normalized_topic = topic.casefold()
        for meeting in meetings:
            meeting_topic = meeting.get("topic")
            if (
                isinstance(meeting_topic, str)
                and meeting_topic.casefold() == normalized_topic
            ):
                return meeting
    return meetings[0] if len(meetings) == 1 else None


def _meeting_has_started(
    runtime: RuntimeContext, start_hour: int | None, start_minute: int | None
) -> bool | None:
    if runtime.hour is None or start_hour is None:
        return None
    current_minutes = runtime.hour * 60 + (runtime.minute or 0)
    start_minutes = start_hour * 60 + (start_minute or 0)
    return current_minutes >= start_minutes


def _record_email_contact_matches(email: EmailFlow, result: Any) -> None:
    if not isinstance(result, dict):
        email.failure_message = "I couldn't look up that contact."
        return

    raw_matches = result.get("matches")
    if not isinstance(raw_matches, dict) or not raw_matches:
        email.failure_message = "I couldn't find that contact."
        return

    matches = {
        str(contact_id): str(name)
        for contact_id, name in raw_matches.items()
        if isinstance(contact_id, str)
    }
    role = email.pending_lookup_role
    if role is None:
        return

    target_name = email.recipient_name if role == "recipient" else email.subject_name
    selected = _select_contact_match_by_target(matches, target_name)
    if selected is None and len(matches) == 1:
        selected = next(iter(matches))

    if selected is None:
        email.pending_contact_matches = matches
        return

    if role == "recipient":
        email.recipient_contact_id = selected
        email.recipient_name = matches[selected]
        _set_email_flow_name(email, "recipient", email.recipient_name)
    else:
        email.subject_contact_id = selected
        email.subject_name = matches[selected]
        _set_email_flow_name(email, "subject", email.subject_name)
    email.pending_lookup_role = None
    email.pending_contact_matches = {}


def _select_contact_match_by_target(
    matches: dict[str, str], target_name: str | None
) -> str | None:
    if not target_name:
        return None
    normalized_target = target_name.casefold()
    for contact_id, name in matches.items():
        if name.casefold() == normalized_target:
            return contact_id
    target_parts = normalized_target.split()
    candidates = []
    for contact_id, name in matches.items():
        normalized_name = name.casefold()
        if all(part in normalized_name.split() for part in target_parts):
            candidates.append(contact_id)
    return candidates[0] if len(candidates) == 1 else None


def _record_email_contact_information(email: EmailFlow, result: Any) -> None:
    if not isinstance(result, dict):
        return
    for contact_id, info in result.items():
        if not isinstance(contact_id, str) or not isinstance(info, dict):
            continue
        if contact_id == email.recipient_contact_id:
            email.recipient_email = _extract_email_from_contact_info(info)
            email.recipient_name = _contact_info_name(info) or email.recipient_name
            if email.recipient_name:
                _set_email_flow_name(email, "recipient", email.recipient_name)
        if contact_id == email.subject_contact_id:
            email.subject_email = _extract_email_from_contact_info(info)
            email.subject_phone = _extract_phone_from_contact_info(info)
            email.subject_name = _contact_info_name(info) or email.subject_name
            if email.subject_name:
                _set_email_flow_name(email, "subject", email.subject_name)


def _extract_email_from_contact_info(info: dict[str, Any]) -> str | None:
    email = info.get("email")
    return email if isinstance(email, str) and email else None


def _extract_phone_from_contact_info(info: dict[str, Any]) -> str | None:
    phone = info.get("phone_number")
    return phone if isinstance(phone, str) and phone else None


def _contact_info_name(info: dict[str, Any]) -> str | None:
    raw_name = info.get("name")
    if isinstance(raw_name, dict):
        first = raw_name.get("first_name")
        last = raw_name.get("last_name")
        parts = [part for part in (first, last) if isinstance(part, str) and part]
        return " ".join(parts) if parts else None
    if isinstance(raw_name, str):
        return raw_name
    return None


def _email_missing_contact_info_ids(email: EmailFlow) -> list[str]:
    ids: list[str] = []
    if email.recipient_contact_id and email.recipient_email is None:
        ids.append(email.recipient_contact_id)
    if (
        email.mode == "share_contact"
        and email.subject_contact_id
        and (email.subject_email is None or email.subject_phone is None)
    ):
        ids.append(email.subject_contact_id)
    return ids


def _calendar_arguments(runtime: RuntimeContext) -> dict[str, Any] | None:
    if runtime.month is None or runtime.day is None:
        return None
    return {"month": runtime.month, "day": runtime.day}


def _build_email_content(state: ControllerState) -> str | None:
    email = state.email
    if email.mode == "meeting_delay":
        recipient_first = email.recipient_first_name or _first_name(
            email.recipient_name
        )
        topic = email.meeting_topic or "our meeting"
        started = _format_clock_time(
            email.meeting_start_hour, email.meeting_start_minute
        )
        delay = _meeting_delay_minutes(state.runtime, email)
        delay_text = (
            f"I'm running about {delay} minutes late"
            if delay is not None and delay > 0
            else "I'm running late"
        )
        return (
            f"Hi {recipient_first}, I wanted to reach out regarding our {topic} "
            f"meeting that started at {started} today. {delay_text} and apologize "
            "for the delay. I should be there shortly. Thank you for your patience. "
            "Best regards"
        )

    if email.mode == "share_contact":
        recipient_first = email.recipient_first_name or _first_name(
            email.recipient_name
        )
        subject_name = email.subject_name
        if not subject_name or not email.subject_phone or not email.subject_email:
            return None
        return (
            f"Hi {recipient_first},\n\n"
            f"I wanted to share {subject_name}'s contact information with you:\n\n"
            f"Name: {subject_name}\n"
            f"Phone: {email.subject_phone}\n"
            f"Email: {email.subject_email}\n\n"
            "Best regards"
        )

    return None


def _format_email_confirmation(email: EmailFlow, extra_recipients: list[str]) -> str:
    addresses = _email_recipient_addresses(email, extra_recipients)
    address_list = ", ".join(addresses)
    return (
        f"I will send this email to {address_list}:\n\n"
        f"{email.content_message}\n\n"
        "Please confirm if I should send it."
    )


def _email_recipient_addresses(
    email: EmailFlow, extra_recipients: list[str]
) -> list[str]:
    addresses = []
    if email.recipient_email:
        addresses.append(email.recipient_email)
    if email.mode == "meeting_delay":
        for extra in extra_recipients:
            if extra not in addresses:
                addresses.append(extra)
    return addresses


def _format_contact_choice_prompt(matches: dict[str, str]) -> str:
    names = ", ".join(name.title() for name in matches.values())
    return f"I found multiple matching contacts: {names}. Which one should I use?"


def _first_name(name: str | None) -> str:
    if not name:
        return "there"
    return name.split()[0]


def _format_clock_time(hour: int | None, minute: int | None) -> str:
    if hour is None:
        return "the scheduled time"
    minute = minute or 0
    return f"{hour:02d}:{minute:02d}"


def _meeting_delay_minutes(runtime: RuntimeContext, email: EmailFlow) -> int | None:
    if runtime.hour is None or email.meeting_start_hour is None:
        return None
    current = runtime.hour * 60 + (runtime.minute or 0)
    start = email.meeting_start_hour * 60 + (email.meeting_start_minute or 0)
    return max(0, current - start)


def _is_sunroof_open_request(text: str) -> bool:
    if "sunroof" not in text:
        return False
    if any(word in text for word in ("close", "shut")):
        return False
    return any(word in text for word in ("open", "fresh air", "vent"))


def _is_sunshade_request(text: str) -> bool:
    if "sunshade" not in text or "sunroof" in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "open",
            "close",
            "shut",
            "adjust",
            "set",
            "help",
            "too bright",
            "sun is",
            "sun's",
        )
    )


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


def _is_window_match_request(text: str) -> bool:
    if not re.search(r"\bwindows?\b", text):
        return False
    if "rear" not in text or "front" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "match",
            "same position",
            "same level",
            "same amount",
            "equal",
            "even",
        )
    )


def _is_ambient_light_request(text: str) -> bool:
    return bool(
        ("ambient" in text and ("light" in text or "lighting" in text))
        or "cabin light" in text
        or "mood light" in text
    )


def _is_ambient_light_match_car_color_request(text: str) -> bool:
    if not _is_ambient_light_request(text):
        return False
    if "match" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "car color",
            "car's color",
            "car colour",
            "car's colour",
            "exterior color",
            "exterior colour",
            "outside color",
            "outside colour",
        )
    )


def _is_light_off_request(text: str) -> bool:
    return (
        _is_close_request(text)
        or "turn off" in text
        or "switch off" in text
        or "deactivate" in text
    )


def _is_high_beam_request(text: str) -> bool:
    if re.search(r"\blow[- ]beams?\b", text):
        return False
    if not (re.search(r"\bhigh[- ]beams?\b", text) or re.search(r"\bbeams?\b", text)):
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "activate",
            "deactivate",
            "set",
            "need",
        )
    )


def _is_fog_light_request(text: str) -> bool:
    if not re.search(r"\bfog[- ]lights?\b", text):
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "activate",
            "deactivate",
            "set",
            "need",
            "want",
        )
    )


def _is_fan_speed_request(text: str) -> bool:
    if not re.search(r"\bfan\b", text):
        return False
    if "airflow direction" in text or "air flow direction" in text:
        return False
    if ("air conditioning" in text or re.search(r"\bac\b", text)) and re.search(
        r"\bwindows?\b", text
    ):
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "set",
            "increase",
            "decrease",
            "level",
            "speed",
        )
    )


def _is_steering_wheel_heating_request(text: str) -> bool:
    if "steering wheel" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "heating",
            "heated",
            "heat",
            "warm",
            "cold",
            "chilly",
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "set",
        )
    )


def _is_reading_light_request(text: str) -> bool:
    if "reading light" not in text and "reading lights" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "activate",
            "deactivate",
            "set",
            "need",
        )
    )


def _is_reading_light_by_occupancy_request(text: str) -> bool:
    if (
        "reading light" not in text
        and "reading lights" not in text
        and "lights" not in text
    ):
        return False
    return any(
        phrase in text
        for phrase in (
            "occupied seats",
            "unoccupied seats",
            "empty seats",
            "who's actually in the car",
            "who is actually in the car",
            "based on who",
            "waste energy",
        )
    )


def _is_defrost_request(text: str) -> bool:
    if "defrost" not in text:
        return False
    if any(phrase in text for phrase in ("match", "same as", "as much as")):
        return False
    if "passenger rear" in text or "driver rear" in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "activate",
            "deactivate",
            "set",
            "clear",
            "fog",
            "fogging",
        )
    )


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


def _is_air_conditioning_request(text: str) -> bool:
    if "air conditioning" not in text and not re.search(r"\bac\b", text):
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "activate",
            "deactivate",
            "set",
            "cool",
        )
    )


def _is_air_conditioning_off_request(text: str) -> bool:
    return bool(
        "deactivate" in text
        or re.search(r"\bturn off (?:the )?(?:air conditioning|ac)\b", text)
        or re.search(r"\bswitch off (?:the )?(?:air conditioning|ac)\b", text)
        or re.search(r"\b(?:air conditioning|ac) off\b", text)
        or re.search(r"\bturn (?:the )?(?:air conditioning|ac) off\b", text)
        or re.search(r"\bswitch (?:the )?(?:air conditioning|ac) off\b", text)
    )


def _is_air_quality_question(text: str) -> bool:
    if not any(word in text for word in ("stagnant", "stale", "stuffy")):
        return False
    if re.search(
        r"\b(turn on|turn off|switch on|switch off|set|activate|deactivate)\b",
        text,
    ):
        return False
    return (
        "what can" in text
        or "what should" in text
        or "current climate settings" in text
        or "climate settings" in text
        or "?" in text
    )


def _is_simple_navigation_request(text: str) -> bool:
    lowered = text.lower()
    if _is_complex_navigation_request(lowered):
        return False
    if _extract_navigation_destination(text) is None:
        return False
    return bool(
        re.search(
            r"\b(navigate|navigation|directions?|route|drive|go|travel)\b",
            lowered,
        )
        or _is_replace_destination_request(lowered)
    )


def _is_navigation_destination_change_followup(
    previous_user_text: str, text: str
) -> bool:
    previous = previous_user_text.lower()
    if not previous:
        return False
    return bool(
        _is_replace_destination_request(previous)
        and re.search(r"\b(go|drive|travel|route|navigate|destination)\b", text)
    )


def _navigation_replace_needs_current_state(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "final destination",
            "multi-stop",
            "multistop",
            "waypoint",
            "intermediate stop",
            " stops ",
            " stop ",
        )
    )


def _is_navigation_delete_waypoint_request(text: str) -> bool:
    if not re.search(r"\b(route|navigation|waypoint|stop)\b", text):
        return False
    return bool(
        re.search(r"\b(remove|delete|skip|drop)\b", text)
        or "no longer need" in text
        or "no longer necessary" in text
        or "without the intermediate stop" in text
    )


def _is_navigation_delete_destination_request(text: str) -> bool:
    if not re.search(r"\b(remove|delete|drop|cancel)\b", text) and not any(
        marker in text
        for marker in (
            "no longer need",
            "don't need",
            "do not need",
            "end my trip at",
            "end the trip at",
        )
    ):
        return False
    return bool(
        "final destination" in text
        or "last stop" in text
        or "final stop" in text
        or re.search(r"\bdelete\b[^.?!]{0,60}\bdestination\b", text)
        or re.search(r"\bcancel\b[^.?!]{0,60}\bdestination\b", text)
        or re.search(r"\bdestination\b[^.?!]{0,60}\b(delete|remove|cancel)\b", text)
    )


def _is_navigation_replace_waypoint_request(text: str) -> bool:
    if "final destination" in text or re.search(r"\bdestination\b", text):
        return False
    if not (
        re.search(r"\b(intermediate stop|waypoint|stop)\b", text)
        or (
            re.search(r"\b(route|navigation)\b", text)
            and re.search(r"\breplace\b[^.?!]{0,80}\bwith\b", text)
        )
    ):
        return False
    return bool(re.search(r"\b(replace|change|switch|swap)\b", text))


def _is_vague_navigation_destination_request(text: str) -> bool:
    if not (
        _is_replace_destination_request(text)
        or re.search(r"\b(navigate|directions?|route|drive|go|travel)\b", text)
    ):
        return False
    vague_markers = (
        "known for",
        "famous for",
        "a city",
        "which city",
        "some city",
        "somewhere",
        "automotive",
    )
    return any(marker in text for marker in vague_markers)


def _location_name_pattern() -> str:
    return r"([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})"


def _extract_navigation_clarified_destination(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:meant|mean|choose|use|destination is|go to|to)\s+{location}",
        rf"^\s*{location}\s*[.?!]?\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            destination = _clean_location_query(value)
            if destination:
                return destination
    return _extract_navigation_destination(text)


def _extract_navigation_new_waypoint(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:with|to)\s+{location}",
        rf"\b(?i:instead|rather)\s+{location}",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            waypoint = _clean_location_query(value)
            if waypoint:
                return waypoint
    return None


def _extract_navigation_waypoint_to_replace(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:replace|swap|change|switch)\s+{location}\s+(?i:with|to)\b",
        rf"\b(?i:from)\s+{location}\s+(?i:to)\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            waypoint = _clean_location_query(value)
            if waypoint and waypoint.lower() not in {"current", "intermediate"}:
                return waypoint
    return None


def _extract_navigation_waypoint_to_delete(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:remove|delete|skip|drop)\s+{location}",
        rf"\b(?i:no longer need)(?: to)? (?i:stop) (?i:in|at)\s+{location}",
        rf"\b{location}\s+(?i:stop|waypoint)\s+(?:(?i:is)\s+)?(?i:no longer)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            waypoint = _clean_location_query(value)
            if waypoint:
                return waypoint
    return None


def _extract_navigation_destination_to_delete(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:remove|delete|drop|cancel)\s+(?:my\s+|the\s+|current\s+)?(?:final\s+destination|destination|final\s+stop|last\s+stop)?\s*(?:\(\s*)?{location}",
        rf"\b(?i:no longer need|don't need|do not need)(?:\s+to\s+(?:go|travel|drive|navigate))?(?:\s+to)?\s+{location}",
        rf"\b{location}\s+(?i:is|as)?\s*(?:my\s+|the\s+)?(?i:final destination|last stop|final stop)[^.?!]{0, 80}\b(?i:remove|delete|drop|cancel|no longer)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            destination = _clean_location_query(value)
            if destination:
                return destination
    return None


def _is_complex_navigation_request(text: str) -> bool:
    complex_markers = (
        "charging station",
        "charging stop",
        "charge",
        "restaurant",
        "fast food",
        "parking",
        "airport",
        "toilet",
        "supermarket",
        "bakery",
        "point of interest",
        "poi",
        "weather",
        "rain",
        "calendar",
        "multi-stop",
        "multistop",
        "waypoint",
        "along the route",
        "stop in",
        "stop at",
        "stops in",
        "stops at",
    )
    return any(marker in text for marker in complex_markers)


def _is_replace_destination_request(text: str) -> bool:
    return bool(
        (
            re.search(r"\b(change|replace|switch|update)\b", text)
            and re.search(r"\b(destination|navigation|route)\b", text)
        )
        or "rather go to" in text
        or "instead of" in text
    )


def _extract_navigation_destination(text: str) -> str | None:
    patterns = (
        r"\b(?i:change|replace|switch|update)\b[^.?!]{0,100}?\b(?i:destination)\b[^.?!]{0,80}?\bto\s+([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})",
        r"\b(?i:change|replace|switch|update)\b[^.?!]{0,100}?\b(?i:navigation|route)\b[^.?!]{0,80}?\bto\s+([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})",
        r"\b(?i:navigate|drive|go|travel|route)\b[^.?!]{0,80}?\bto\s+([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})",
        r"\b(?i:directions?)\b[^.?!]{0,40}?\bto\s+([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})",
        r"\b(?i:rather go to)\s+([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            destination = _clean_location_query(matches[-1])
            if destination:
                return destination

    return None


def _clean_location_query(value: str) -> str | None:
    value = value.split(",", 1)[0]
    value = re.split(
        r"\b(?:instead|because|with|without|if|when|that|which|where|for|from|via|then|first|next|now|rather)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.sub(
        r"\b(?:city center|city centre|city|centre)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = " ".join(value.strip(" .?!;:-").split())
    value = re.sub(r"\s+\b(?:and|or)\b$", "", value, flags=re.IGNORECASE)
    if not value:
        return None
    if value.lower() in {"the", "there", "home", "work"}:
        return None
    return value


def _matching_recent_location_id(
    requested_name: str | None,
    candidate_name: str | None,
    candidate_id: str | None,
) -> str | None:
    if not requested_name or not candidate_name or not candidate_id:
        return None
    if requested_name.casefold() == candidate_name.casefold():
        return candidate_id
    return None


def _extract_route_preference(
    text: str,
) -> Literal["fastest", "shortest"] | None:
    if "fastest" in text or "quickest" in text:
        return "fastest"
    if "shortest" in text:
        return "shortest"
    return None


def _extract_route_choice_index(text: str) -> int | None:
    ordinal_to_index = {
        "first": 0,
        "1st": 0,
        "one": 0,
        "second": 1,
        "2nd": 1,
        "two": 1,
        "third": 2,
        "3rd": 2,
        "three": 2,
    }
    if match := re.search(r"\b(?:route|option)\s+([123])\b", text):
        return int(match.group(1)) - 1
    if match := re.search(r"\b([123])(?:st|nd|rd)?\s+(?:route|option)\b", text):
        return int(match.group(1)) - 1
    for word, index in ordinal_to_index.items():
        if re.search(rf"\b{word}\s+(?:route|option)\b", text) or re.search(
            rf"\b(?:route|option)\s+{word}\b", text
        ):
            return index
    if re.search(r"\b(another|different|alternative|other)\s+(?:route|option)\b", text):
        return 1
    return None


def _navigation_route_start_id(
    state: ControllerState,
    navigation: NavigationFlow,
) -> str | None:
    if navigation.route_start_id:
        return navigation.route_start_id
    if (
        navigation.mode == "replace_final_destination"
        and len(navigation.waypoints_id) >= 2
    ):
        return navigation.waypoints_id[-2]
    return state.runtime.location_id


def _record_navigation_routes(
    navigation: NavigationFlow, routes: list[dict[str, Any]]
) -> None:
    if navigation.route_lookup == "delete_without_waypoint":
        navigation.route_without_waypoint_checked = True
        navigation.routes_without_waypoint = routes
    elif navigation.route_lookup == "to_new_waypoint":
        navigation.route_to_new_waypoint_checked = True
        navigation.routes_to_new_waypoint = routes
    elif navigation.route_lookup == "from_new_waypoint":
        navigation.route_from_new_waypoint_checked = True
        navigation.routes_from_new_waypoint = routes
    else:
        navigation.routes_checked = True
        navigation.routes = routes


def _navigation_choice_routes(navigation: NavigationFlow) -> list[dict[str, Any]]:
    if navigation.route_lookup == "delete_without_waypoint":
        return navigation.routes_without_waypoint
    if navigation.route_lookup == "to_new_waypoint":
        return navigation.routes_to_new_waypoint
    if navigation.route_lookup == "from_new_waypoint":
        return navigation.routes_from_new_waypoint
    return navigation.routes


def _resolve_navigation_waypoint_from_state(navigation: NavigationFlow) -> None:
    if navigation.waypoint_id:
        return

    waypoint_name = navigation.waypoint_name
    if waypoint_name:
        normalized = waypoint_name.casefold()
        for waypoint in navigation.waypoint_details:
            name = waypoint.get("name")
            waypoint_id = waypoint.get("id")
            if (
                isinstance(name, str)
                and isinstance(waypoint_id, str)
                and name.casefold() == normalized
            ):
                navigation.waypoint_id = waypoint_id
                return
        return

    intermediate_waypoints = navigation.waypoints_id[1:-1]
    if len(intermediate_waypoints) == 1:
        navigation.waypoint_id = intermediate_waypoints[0]


def _resolve_navigation_final_destination_from_state(
    navigation: NavigationFlow,
) -> None:
    if navigation.destination_id:
        return
    if not navigation.waypoints_id:
        return

    final_destination_id = navigation.waypoints_id[-1]
    if not navigation.destination_name:
        navigation.destination_id = final_destination_id
        return

    normalized = navigation.destination_name.casefold()
    for waypoint in navigation.waypoint_details:
        name = waypoint.get("name")
        waypoint_id = waypoint.get("id")
        if (
            isinstance(name, str)
            and isinstance(waypoint_id, str)
            and name.casefold() == normalized
            and waypoint_id == final_destination_id
        ):
            navigation.destination_id = waypoint_id
            return


def _navigation_adjacent_waypoints(
    navigation: NavigationFlow, waypoint_id: str
) -> tuple[str, str] | None:
    try:
        waypoint_index = navigation.waypoints_id.index(waypoint_id)
    except ValueError:
        return None

    if waypoint_index <= 0 or waypoint_index >= len(navigation.waypoints_id) - 1:
        return None

    return (
        navigation.waypoints_id[waypoint_index - 1],
        navigation.waypoints_id[waypoint_index + 1],
    )


def _navigation_completion_message(
    action_summary: str,
    route: dict[str, Any],
    preference: Literal["fastest", "shortest"] | None,
) -> str:
    parts = [f"Done, {action_summary}."]
    route_choice = _route_choice_label(route, preference)
    if route_choice:
        parts.append(f"I used the {route_choice} route.")
    if _route_has_toll(route):
        parts.append("This route includes toll roads.")
    if route_choice:
        parts.append("Would you like information on alternative routes?")
    return " ".join(parts)


def _navigation_multi_route_completion_message(
    action_summary: str,
    routes: list[dict[str, Any]],
    preference: Literal["fastest", "shortest"] | None,
) -> str:
    parts = [f"Done, {action_summary}."]
    route_choice = _route_choice_label(routes[0], preference) if routes else None
    if route_choice:
        parts.append(f"I used the {route_choice} route segments.")
    if any(_route_has_toll(route) for route in routes):
        parts.append("At least one route segment includes toll roads.")
    if route_choice:
        parts.append("Would you like information on alternative route segments?")
    return " ".join(parts)


def _route_choice_label(
    route: dict[str, Any], preference: Literal["fastest", "shortest"] | None
) -> str | None:
    if preference is not None:
        return preference
    aliases = route.get("alias")
    if isinstance(aliases, list):
        lowered_aliases = {str(alias).lower() for alias in aliases}
        if "fastest" in lowered_aliases and "shortest" in lowered_aliases:
            return "fastest and shortest"
        if "fastest" in lowered_aliases:
            return "fastest"
        if "shortest" in lowered_aliases:
            return "shortest"
        if "second" in lowered_aliases:
            return "second"
        if "third" in lowered_aliases:
            return "third"
        if "first" in lowered_aliases:
            return "first"
    return None


def _route_has_toll(route: dict[str, Any]) -> bool:
    if route.get("includes_toll") is True:
        return True
    road_types = route.get("road_types")
    if isinstance(road_types, list):
        return any("toll" in str(road_type).lower() for road_type in road_types)
    return False


def _select_route(
    routes: list[dict[str, Any]],
    preference: Literal["fastest", "shortest"] | None,
) -> dict[str, Any] | None:
    if not routes:
        return None
    if preference is None:
        return routes[0] if len(routes) == 1 else None
    for route in routes:
        aliases = route.get("alias") or []
        if isinstance(aliases, list) and preference in {
            str(alias).lower() for alias in aliases
        }:
            return route
    return routes[0] if len(routes) == 1 else None


def _select_requested_route(
    routes: list[dict[str, Any]],
    preference: Literal["fastest", "shortest"] | None,
    selected_index: int | None,
) -> dict[str, Any] | None:
    if selected_index is not None:
        if 0 <= selected_index < len(routes):
            return routes[selected_index]
        return None
    return _select_route(routes, preference)


def _select_route_by_alias(
    routes: list[dict[str, Any]], alias: str
) -> dict[str, Any] | None:
    for route in routes:
        aliases = route.get("alias") or []
        if isinstance(aliases, list) and alias in {
            str(route_alias).lower() for route_alias in aliases
        }:
            return route
    return None


def _route_choice_is_unambiguous(routes: list[dict[str, Any]]) -> bool:
    if len(routes) == 1:
        return True
    fastest = _select_route(routes, "fastest")
    shortest = _select_route(routes, "shortest")
    return (
        fastest is not None
        and shortest is not None
        and fastest.get("route_id") == shortest.get("route_id")
    )


def _format_route_choice_prompt(routes: list[dict[str, Any]]) -> str:
    fastest = _select_route(routes, "fastest")
    shortest = _select_route(routes, "shortest")
    if (
        fastest is not None
        and shortest is not None
        and fastest.get("route_id") == shortest.get("route_id")
    ):
        second_route = _select_route_by_alias(routes, "second")
        if second_route is None and len(routes) > 1:
            second_route = routes[1]
        if second_route is not None and second_route.get("route_id") != fastest.get(
            "route_id"
        ):
            return (
                f"I found routes. Fastest and shortest: {_route_summary(fastest)}. "
                f"Second route: {_route_summary(second_route)}. "
                "Do you want the first route or the second route?"
            )
        return (
            f"I selected the fastest route: {_route_summary(fastest)}. "
            "It is also the shortest route. Do you want me to apply it, "
            "or would you like more information on alternative routes?"
        )
    if fastest is not None and shortest is not None:
        extra_count = max(len(routes) - len({id(fastest), id(shortest)}), 0)
        extra_note = (
            f" There are {extra_count} other alternatives." if extra_count else ""
        )
        return (
            f"I found routes. Fastest: {_route_summary(fastest)}. "
            f"Shortest: {_route_summary(shortest)}.{extra_note} "
            "Do you want the fastest or shortest route?"
        )
    if len(routes) > 1:
        second_route = _select_route_by_alias(routes, "second") or routes[1]
        return (
            f"I found routes. First route: {_route_summary(routes[0])}. "
            f"Second route: {_route_summary(second_route)}. "
            "Do you want the first route or the second route?"
        )
    return "I found multiple routes. Do you want the fastest or shortest route?"


def _route_summary(route: dict[str, Any]) -> str:
    distance = route.get("distance_km")
    hours = _safe_int(route.get("duration_hours"), 0) or 0
    minutes = _safe_int(route.get("duration_minutes"), 0) or 0
    via = route.get("name_via")
    parts = []
    if isinstance(via, str) and via:
        parts.append(f"via {via}")
    if isinstance(distance, (int, float)):
        parts.append(f"{distance:g} km")
    if hours or minutes:
        parts.append(_format_duration(hours, minutes))
    if _route_has_toll(route):
        parts.append("includes toll roads")
    return ", ".join(parts) if parts else "route details are available"


def _format_duration(hours: int, minutes: int) -> str:
    if hours and minutes:
        return f"{hours} h {minutes} min"
    if hours:
        return f"{hours} h"
    return f"{minutes} min"


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


def _extract_sunshade_percentage(text: str) -> int | None:
    return _extract_percentage(text)


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
    if "sunshade" in text and re.search(
        r"sunshade[^.?!,;]*(fully|all the way|100)", text
    ):
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


def _extract_level(text: str) -> int | None:
    match = re.search(r"\blevel\s*([0-5])\b", text)
    if match:
        return int(match.group(1))

    match = re.search(r"\bfan[^.?!,;]*?\b([0-5])\b", text)
    if match:
        return int(match.group(1))

    return None


def _extract_heating_level(text: str) -> int | None:
    level = _extract_level(text)
    if level is not None:
        return level
    if "medium" in text:
        return 2
    if "low" in text:
        return 1
    if "high" in text:
        return 3
    return None


def _extract_reading_light_position(text: str) -> str | None:
    if "driver rear" in text or "rear driver" in text or "left rear" in text:
        return "DRIVER_REAR"
    if "passenger rear" in text or "rear passenger" in text or "right rear" in text:
        return "PASSENGER_REAR"
    if re.search(r"\bdriver\b", text) and "rear" not in text:
        return "DRIVER"
    if re.search(r"\bpassenger\b", text) and "rear" not in text:
        return "PASSENGER"
    if re.search(r"\b(all|both)\b", text):
        return "ALL"
    return None


def _extract_air_circulation_mode(
    text: str,
) -> Literal["FRESH_AIR", "RECIRCULATION", "AUTO"] | None:
    if "fresh air" in text or "outside air" in text:
        return "FRESH_AIR"
    if "recirculation" in text or "recirculate" in text:
        if any(
            phrase in text
            for phrase in ("don't like", "do not like", "not recirculation")
        ):
            return None
        return "RECIRCULATION"
    if re.search(r"\bauto(?:matic)?\b", text):
        return "AUTO"
    return None


def _extract_defrost_window(
    text: str,
) -> Literal["ALL", "FRONT", "REAR"] | None:
    if "front" in text or "windshield" in text or "windscreen" in text:
        return "FRONT"
    if re.search(r"\brear(?: window)? defrost\b", text) or re.search(
        r"\bdefrost(?: the)? rear\b", text
    ):
        return "REAR"
    if re.search(r"\b(all|both) (?:window )?defrost\b", text) or re.search(
        r"\bdefrost (?:all|both)\b", text
    ):
        return "ALL"
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


def _is_negative(text: str) -> bool:
    text = text.strip().lower()
    return bool(
        re.search(r"\b(no|nope|nah|cancel|stop|decline|declined)\b", text)
        or "don't" in text
        or "do not" in text
    )


def _tool_result_name(tool_result: dict[str, Any]) -> str:
    return (
        tool_result.get("tool_name")
        or tool_result.get("toolName")
        or tool_result.get("name")
        or ""
    )


def _parse_tool_call_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _tool_argument_available(
    tool_index: ToolIndex, tool_name: str, argument_name: str
) -> bool:
    return argument_name in tool_index.arg_names(tool_name)


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


def _extract_preferred_fan_speed_level(result: dict[str, Any]) -> int | None:
    text = json.dumps(result, ensure_ascii=False).lower()
    if "fan" not in text:
        return None

    patterns = (
        r"fan[^.?!;]*?level\s*([0-5])",
        r"fan[^.?!;]*?speed[^.?!;]*?([0-5])",
        r"level\s*([0-5])[^.?!;]*?fan",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _safe_int(match.group(1))
    return None


def _extract_preferred_steering_heating_level(result: dict[str, Any]) -> int | None:
    text = json.dumps(result, ensure_ascii=False).lower()
    if "steering wheel" not in text:
        return None

    match = re.search(r"steering wheel[^.?!;]*?level\s*([0-5])", text)
    if match:
        return _safe_int(match.group(1))

    match = re.search(r"level\s*([0-5])[^.?!;]*?steering wheel", text)
    if match:
        return _safe_int(match.group(1))

    if "medium" in text:
        return 2
    if "low" in text:
        return 1
    if "high" in text:
        return 3
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


def _record_defrost_window_position(
    defrost: DefrostFlow, window: str, percentage: int
) -> None:
    if window == "ALL":
        defrost.window_driver_position = percentage
        defrost.window_passenger_position = percentage
        defrost.window_driver_rear_position = percentage
        defrost.window_passenger_rear_position = percentage
    elif window == "DRIVER":
        defrost.window_driver_position = percentage
    elif window == "PASSENGER":
        defrost.window_passenger_position = percentage
    elif window in {"DRIVER_REAR", "RIGHT_REAR"}:
        defrost.window_driver_rear_position = percentage
    elif window in {"PASSENGER_REAR", "LEFT_REAR"}:
        defrost.window_passenger_rear_position = percentage


def _record_window_match_position(
    match: WindowMatchFlow, window: str, percentage: int
) -> None:
    if window == "ALL":
        match.window_driver_position = percentage
        match.window_passenger_position = percentage
        match.window_driver_rear_position = percentage
        match.window_passenger_rear_position = percentage
    elif window == "DRIVER":
        match.window_driver_position = percentage
    elif window == "PASSENGER":
        match.window_passenger_position = percentage
    elif window in {"DRIVER_REAR", "RIGHT_REAR"}:
        match.window_driver_rear_position = percentage
    elif window in {"PASSENGER_REAR", "LEFT_REAR"}:
        match.window_passenger_rear_position = percentage


def _defrost_window_positions(defrost: DefrostFlow) -> tuple[int | None, ...]:
    return (
        defrost.window_driver_position,
        defrost.window_passenger_position,
        defrost.window_driver_rear_position,
        defrost.window_passenger_rear_position,
    )


def _defrost_window_status_unknown(defrost: DefrostFlow) -> bool:
    return any(value is None for value in _defrost_window_positions(defrost))


def _record_air_conditioning_window_position(
    ac: AirConditioningFlow, window: str, percentage: int
) -> None:
    if window == "ALL":
        ac.window_driver_position = percentage
        ac.window_passenger_position = percentage
        ac.window_driver_rear_position = percentage
        ac.window_passenger_rear_position = percentage
    elif window == "DRIVER":
        ac.window_driver_position = percentage
    elif window == "PASSENGER":
        ac.window_passenger_position = percentage
    elif window in {"DRIVER_REAR", "RIGHT_REAR"}:
        ac.window_driver_rear_position = percentage
    elif window in {"PASSENGER_REAR", "LEFT_REAR"}:
        ac.window_passenger_rear_position = percentage


def _air_conditioning_window_positions(
    ac: AirConditioningFlow,
) -> tuple[tuple[str, int | None], ...]:
    return (
        ("DRIVER", ac.window_driver_position),
        ("PASSENGER", ac.window_passenger_position),
        ("DRIVER_REAR", ac.window_driver_rear_position),
        ("PASSENGER_REAR", ac.window_passenger_rear_position),
    )


def _air_conditioning_window_status_unknown(ac: AirConditioningFlow) -> bool:
    return any(
        position is None for _, position in _air_conditioning_window_positions(ac)
    )


def _air_conditioning_open_windows(ac: AirConditioningFlow) -> list[tuple[str, int]]:
    return [
        (window, position)
        for window, position in _air_conditioning_window_positions(ac)
        if position is not None and position > 20
    ]


def _reading_light_actions_for_occupancy(
    seats_occupied: dict[str, bool],
) -> list[tuple[str, bool]]:
    seat_to_light = {
        "driver": "DRIVER",
        "passenger": "PASSENGER",
        "driver_rear": "DRIVER_REAR",
        "passenger_rear": "PASSENGER_REAR",
    }
    occupied_order = ("driver", "passenger", "driver_rear", "passenger_rear")
    unoccupied_order = ("driver_rear", "passenger_rear", "passenger", "driver")

    actions: list[tuple[str, bool]] = []
    for seat in occupied_order:
        if seats_occupied.get(seat) is True:
            actions.append((seat_to_light[seat], True))
    for seat in unoccupied_order:
        if seats_occupied.get(seat) is False:
            actions.append((seat_to_light[seat], False))
    return actions


def _any_defrost_window_open_over(defrost: DefrostFlow, threshold: int) -> bool:
    return any(
        value is not None and value > threshold
        for value in _defrost_window_positions(defrost)
    )


def _airflow_includes_windshield(direction: str | None) -> bool:
    return bool(direction and "WINDSHIELD" in direction)


def _friendly_defrost_window(
    defrost_window: Literal["ALL", "FRONT", "REAR"] | None,
) -> str:
    return {
        "ALL": "all-window",
        "FRONT": "front-window",
        "REAR": "rear-window",
        None: "window",
    }[defrost_window]


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


def _friendly_reading_light_position(position: str | None) -> str:
    return {
        "ALL": "all",
        "DRIVER": "the driver",
        "PASSENGER": "the passenger",
        "DRIVER_REAR": "the driver rear",
        "PASSENGER_REAR": "the passenger rear",
        None: "the requested",
    }.get(position, "the requested")


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


def _requires_fog_light_weather_confirmation(condition: str | None) -> bool:
    if not condition:
        return True
    return condition not in SAFE_FOG_LIGHT_WEATHER


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
