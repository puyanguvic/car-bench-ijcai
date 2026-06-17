import json
from typing import Any

from carbench_agent_core import PolicyAwareController


def fake_tool(
    name: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


def tool_result(tool_name: str, result: dict) -> dict:
    return {
        "tool_name": tool_name,
        "content": json.dumps({"status": "SUCCESS", "result": result}),
    }


def messages() -> list[dict]:
    return [{"role": "system", "content": ""}]


def climate_tools(*, include_fan_level: bool = True) -> list[dict]:
    fan_properties = {"level": {"type": "integer"}} if include_fan_level else {}
    fan_required = ["level"] if include_fan_level else []
    return [
        fake_tool("get_climate_settings"),
        fake_tool("get_vehicle_window_positions"),
        fake_tool(
            "open_close_window",
            {
                "window": {"type": "string"},
                "percentage": {"type": "number"},
            },
            ["window", "percentage"],
        ),
        fake_tool("set_fan_speed", fan_properties, fan_required),
        fake_tool("set_air_conditioning", {"on": {"type": "boolean"}}, ["on"]),
    ]


def climate_result(*, fan_speed: int = 0, air_conditioning: bool = False) -> dict:
    return {
        "fan_speed": fan_speed,
        "fan_airflow_direction": "HEAD",
        "air_conditioning": air_conditioning,
        "air_circulation": "FRESH_AIR",
    }


def window_positions(
    *,
    driver: int = 0,
    passenger: int = 0,
    driver_rear: int = 0,
    passenger_rear: int = 0,
) -> dict:
    return {
        "window_driver_position": driver,
        "window_passenger_position": passenger,
        "window_driver_rear_position": driver_rear,
        "window_passenger_rear_position": passenger_rear,
    }


def test_air_conditioning_controller_satisfies_policy_before_turning_on() -> None:
    controller = PolicyAwareController()
    tools = climate_tools()

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the air conditioning.",
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_vehicle_window_positions", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_vehicle_window_positions",
                window_positions(driver=100, passenger=10, passenger_rear=25),
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "DRIVER", "percentage": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "DRIVER", "percentage": 0})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "PASSENGER_REAR", "percentage": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "open_close_window",
                {"window": "PASSENGER_REAR", "percentage": 0},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 1}}
    ]

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_fan_speed", {"level": 1})],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
    ]


def test_air_conditioning_controller_preserves_explicit_fan_level() -> None:
    controller = PolicyAwareController()
    tools = climate_tools()

    controller.decide(
        context_id="ctx-ac-explicit-fan",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Close the window fully open, turn on the air conditioning, "
            "and set the fan speed to level 3."
        ),
    )
    controller.decide(
        context_id="ctx-ac-explicit-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    action = controller.decide(
        context_id="ctx-ac-explicit-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_vehicle_window_positions",
                window_positions(driver=5, driver_rear=100),
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "DRIVER_REAR", "percentage": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-ac-explicit-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "DRIVER_REAR", "percentage": 0})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 3}}
    ]


def test_air_conditioning_controller_refuses_when_fan_level_parameter_is_missing() -> None:
    controller = PolicyAwareController()
    tools = climate_tools(include_fan_level=False)

    controller.decide(
        context_id="ctx-ac-missing-fan-level",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the AC.",
    )
    controller.decide(
        context_id="ctx-ac-missing-fan-level",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    action = controller.decide(
        context_id="ctx-ac-missing-fan-level",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_vehicle_window_positions", window_positions())
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fan speed level control is unavailable" in action.content.lower()


def test_air_quality_question_clarifies_before_taking_action() -> None:
    controller = PolicyAwareController()
    tools = climate_tools()

    action = controller.decide(
        context_id="ctx-air-quality",
        messages=messages(),
        tools=tools,
        latest_user_text="The air in the car feels a bit stagnant. What can be done about that?",
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-air-quality",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fan is currently off" in action.content.lower()
    assert "which would you like" in action.content.lower()

    action = controller.decide(
        context_id="ctx-air-quality",
        messages=messages(),
        tools=tools,
        latest_user_text="Set it to level 2.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}}
    ]


def test_sunshade_controller_asks_before_setting_ambiguous_percentage() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool(
            "open_close_sunshade",
            {"percentage": {"type": "number"}},
            ["percentage"],
        )
    ]

    action = controller.decide(
        context_id="ctx-sunshade",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "The sun is really bright this morning. Can you help me with the sunshade?"
        ),
    )
    assert action is not None
    assert action.action == "respond"
    assert "how far" in action.content.lower()

    action = controller.decide(
        context_id="ctx-sunshade",
        messages=messages(),
        tools=tools,
        latest_user_text="Let's go with 50%.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "open_close_sunshade", "arguments": {"percentage": 50}}
    ]


def test_fan_speed_controller_uses_preference_before_setting_level() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_user_preferences"),
        fake_tool("set_fan_speed", {"level": {"type": "integer"}}, ["level"]),
    ]

    action = controller.decide(
        context_id="ctx-fan-pref",
        messages=messages(),
        tools=tools,
        latest_user_text="Could you turn on the fan?",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_user_preferences",
            "arguments": {
                "preference_categories": {
                    "vehicle_settings": {"climate_control": True}
                }
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-fan-pref",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_user_preferences",
                {
                    "vehicle_settings": {
                        "climate_control": [
                            "user prefers fan speed level 3 as default value for moderate airflow"
                        ]
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 3}}
    ]


def test_fan_speed_controller_asks_when_preference_is_missing() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_user_preferences"),
        fake_tool("set_fan_speed", {"level": {"type": "integer"}}, ["level"]),
    ]

    controller.decide(
        context_id="ctx-fan-ask",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the fan.",
    )
    action = controller.decide(
        context_id="ctx-fan-ask",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_user_preferences", {})],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fan speed level" in action.content.lower()

    action = controller.decide(
        context_id="ctx-fan-ask",
        messages=messages(),
        tools=tools,
        latest_user_text="Set it to level 3.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 3}}
    ]


def test_fan_speed_controller_refuses_when_level_parameter_is_missing() -> None:
    controller = PolicyAwareController()
    tools = [fake_tool("set_fan_speed")]

    action = controller.decide(
        context_id="ctx-fan-missing-level",
        messages=messages(),
        tools=tools,
        latest_user_text="Set the fan speed to level 3.",
    )
    assert action is not None
    assert action.action == "respond"
    assert "level control is unavailable" in action.content.lower()


def test_steering_wheel_heating_controller_uses_preference_level() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_user_preferences"),
        fake_tool(
            "set_steering_wheel_heating",
            {"level": {"type": "integer"}},
            ["level"],
        ),
    ]

    action = controller.decide(
        context_id="ctx-heating-pref",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the steering wheel heating.",
    )
    assert action is not None
    assert action.tool_calls[0]["tool_name"] == "get_user_preferences"

    action = controller.decide(
        context_id="ctx-heating-pref",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_user_preferences",
                {
                    "vehicle_settings": {
                        "vehicle_settings": [
                            "If steering wheel heating should be turned on, user prefers level 2, because level 1 is not feelable and level 3 is too hot"
                        ]
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_steering_wheel_heating", "arguments": {"level": 2}}
    ]


def test_reading_light_controller_asks_for_ambiguous_position() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool(
            "set_reading_light",
            {
                "position": {"type": "string"},
                "on": {"type": "boolean"},
            },
            ["position", "on"],
        )
    ]

    action = controller.decide(
        context_id="ctx-reading-light",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the reading lights.",
    )
    assert action is not None
    assert action.action == "respond"
    assert "which reading light" in action.content.lower()

    action = controller.decide(
        context_id="ctx-reading-light",
        messages=messages(),
        tools=tools,
        latest_user_text="The driver reading light.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_reading_light",
            "arguments": {"position": "DRIVER", "on": True},
        }
    ]


def test_reading_light_controller_refuses_when_position_parameter_is_missing() -> None:
    controller = PolicyAwareController()
    tools = [fake_tool("set_reading_light", {"on": {"type": "boolean"}}, ["on"])]

    action = controller.decide(
        context_id="ctx-reading-light-missing-position",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the driver reading light.",
    )
    assert action is not None
    assert action.action == "respond"
    assert "position control is unavailable" in action.content.lower()


def test_high_beam_generic_beam_request_checks_exterior_status() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_exterior_lights_status"),
        fake_tool("set_head_lights_high_beams", {"on": {"type": "boolean"}}, ["on"]),
    ]

    action = controller.decide(
        context_id="ctx-generic-beams",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the beams, please.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_exterior_lights_status", "arguments": {}}
    ]


def test_high_beam_controller_refuses_when_fog_status_is_missing() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_exterior_lights_status"),
        fake_tool("set_head_lights_high_beams", {"on": {"type": "boolean"}}, ["on"]),
    ]

    controller.decide(
        context_id="ctx-high-beams-missing-fog",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the high beams.",
    )
    action = controller.decide(
        context_id="ctx-high-beams-missing-fog",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_exterior_lights_status",
                {
                    "head_lights_low_beams": True,
                    "head_lights_high_beams": False,
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fog light status" in action.content.lower()
