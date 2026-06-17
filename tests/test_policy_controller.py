import json

from carbench_agent_core import PolicyAwareController


def fake_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def tool_result(tool_name: str, result: dict) -> dict:
    return {
        "tool_name": tool_name,
        "content": json.dumps({"status": "SUCCESS", "result": result}),
    }


def system_messages(location_id: str = "loc_bonn_001") -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                f'CURRENT_LOCATION = {{"id": "{location_id}", "name": "Bonn"}}\n'
                'DATETIME = {"year": 2026, "month": 6, "day": 17, '
                '"hour": 9, "minute": 0}'
            ),
        }
    ]


def navigation_tools() -> list[dict]:
    return [
        fake_tool("get_location_id_by_location_name"),
        fake_tool("get_routes_from_start_to_destination"),
        fake_tool("set_new_navigation"),
        fake_tool("navigation_replace_final_destination"),
    ]


def test_navigation_controller_sets_fastest_single_destination_route() -> None:
    controller = PolicyAwareController()
    messages = system_messages()
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-nav",
        messages=messages,
        tools=tools,
        latest_user_text="Please navigate to Munich on the fastest route.",
    )
    assert action is not None
    assert action.action == "tool_calls"
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        }
    ]

    action = controller.decide(
        context_id="ctx-nav",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_mun_002"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_bonn_001",
                "destination_id": "loc_mun_002",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-nav",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {
                            "route_id": "rll_bonn_mun_fast",
                            "alias": ["first", "fastest"],
                            "distance_km": 520,
                            "duration_hours": 5,
                            "duration_minutes": 30,
                        },
                        {
                            "route_id": "rll_bonn_mun_short",
                            "alias": ["second", "shortest"],
                            "distance_km": 500,
                            "duration_hours": 6,
                            "duration_minutes": 15,
                        },
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_new_navigation",
            "arguments": {"route_ids": ["rll_bonn_mun_fast"]},
        }
    ]


def test_navigation_controller_asks_when_route_choice_is_ambiguous() -> None:
    controller = PolicyAwareController()
    messages = system_messages()
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-choice",
        messages=messages,
        tools=tools,
        latest_user_text="Please navigate to Munich.",
    )
    controller.decide(
        context_id="ctx-choice",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_mun_002"})
        ],
    )

    action = controller.decide(
        context_id="ctx-choice",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {
                            "route_id": "rll_bonn_mun_fast",
                            "alias": ["fastest"],
                            "distance_km": 520,
                            "duration_hours": 5,
                            "duration_minutes": 30,
                        },
                        {
                            "route_id": "rll_bonn_mun_short",
                            "alias": ["shortest"],
                            "distance_km": 500,
                            "duration_hours": 6,
                            "duration_minutes": 15,
                            "includes_toll": True,
                        },
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fastest" in action.content.lower()
    assert "shortest" in action.content.lower()
    assert "toll" in action.content.lower()

    action = controller.decide(
        context_id="ctx-choice",
        messages=messages,
        tools=tools,
        latest_user_text="Take the shortest route.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_new_navigation",
            "arguments": {"route_ids": ["rll_bonn_mun_short"]},
        }
    ]


def test_navigation_controller_replaces_final_destination_with_route_evidence() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_bochum_001")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-replace",
        messages=messages,
        tools=tools,
        latest_user_text="Change my navigation destination from Milan to Hamburg.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Hamburg"},
        }
    ]

    action = controller.decide(
        context_id="ctx-replace",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_ham_003"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_bochum_001",
                "destination_id": "loc_ham_003",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-replace",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [{"route_id": "rll_bochum_ham", "alias": ["fastest"]}]},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": "loc_ham_003",
                "route_id_leading_to_new_destination": "rll_bochum_ham",
            },
        }
    ]


def test_navigation_controller_does_not_take_over_complex_poi_routes() -> None:
    controller = PolicyAwareController()

    action = controller.decide(
        context_id="ctx-complex",
        messages=system_messages(),
        tools=navigation_tools(),
        latest_user_text=(
            "Set up navigation to Andorra la Vella with a charging station stop."
        ),
    )

    assert action is None
