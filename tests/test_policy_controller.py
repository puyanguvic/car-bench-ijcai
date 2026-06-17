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
        fake_tool("get_current_navigation_state"),
        fake_tool("get_location_id_by_location_name"),
        fake_tool("get_routes_from_start_to_destination"),
        fake_tool("set_new_navigation"),
        fake_tool("navigation_replace_final_destination"),
        fake_tool("navigation_delete_waypoint"),
        fake_tool("navigation_replace_one_waypoint"),
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


def test_navigation_followup_replaces_destination_after_change_prompt() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_bochum_001")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-replace-followup",
        messages=messages,
        tools=tools,
        latest_user_text="I need to change my navigation destination. Can you help?",
    )
    assert action is None

    action = controller.decide(
        context_id="ctx-replace-followup",
        messages=messages,
        tools=tools,
        latest_user_text="I need to go to Hamburg now.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Hamburg"},
        }
    ]

    controller.decide(
        context_id="ctx-replace-followup",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_ham_003"})
        ],
    )
    action = controller.decide(
        context_id="ctx-replace-followup",
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


def test_navigation_controller_replaces_final_destination_from_route_predecessor() -> (
    None
):
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_andorra")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-final-predecessor",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Change my final destination from Rome to Stuttgart and use the "
            "shortest route."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Stuttgart"},
        }
    ]

    action = controller.decide(
        context_id="ctx-final-predecessor",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_stuttgart"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": False},
        }
    ]

    action = controller.decide(
        context_id="ctx-final-predecessor",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_andorra",
                        "loc_paris",
                        "loc_milan",
                        "loc_rome",
                    ],
                    "routes_to_final_destination_id": [
                        "rll_and_par",
                        "rll_par_mil",
                        "rll_mil_rom",
                    ],
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_milan",
                "destination_id": "loc_stuttgart",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-final-predecessor",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "rll_mil_stu_fast", "alias": ["fastest"]},
                        {"route_id": "rll_mil_stu_short", "alias": ["shortest"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": "loc_stuttgart",
                "route_id_leading_to_new_destination": "rll_mil_stu_short",
            },
        }
    ]


def test_navigation_followup_reuses_observed_waypoints_after_external_route_edit() -> (
    None
):
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_andorra")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-followup-observed-waypoints",
        messages=messages,
        tools=tools,
        latest_user_text="I need to change my final destination.",
    )
    controller.decide(
        context_id="ctx-followup-observed-waypoints",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "navigation_replace_final_destination",
                {
                    "destination_replaced": True,
                    "new_waypoints": [
                        "loc_andorra",
                        "loc_paris",
                        "loc_milan",
                        "loc_stuttgart",
                    ],
                    "new_routes": [
                        "rll_and_par",
                        "rll_par_mil",
                        "rll_mil_stu",
                    ],
                },
            )
        ],
    )

    action = controller.decide(
        context_id="ctx-followup-observed-waypoints",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Oh, I meant Munich. Can you find the shortest route to Munich instead?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        }
    ]

    action = controller.decide(
        context_id="ctx-followup-observed-waypoints",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_munich"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_milan",
                "destination_id": "loc_munich",
            },
        }
    ]


def test_navigation_followup_checks_current_state_when_waypoints_are_unknown() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_andorra")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-followup-current-state",
        messages=messages,
        tools=tools,
        latest_user_text="I need to change my final destination.",
    )

    action = controller.decide(
        context_id="ctx-followup-current-state",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Oh, I meant Munich. Can you find the shortest route to Munich instead?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        }
    ]

    action = controller.decide(
        context_id="ctx-followup-current-state",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_munich"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": False},
        }
    ]

    action = controller.decide(
        context_id="ctx-followup-current-state",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_andorra",
                        "loc_paris",
                        "loc_milan",
                        "loc_rome",
                    ],
                    "routes_to_final_destination_id": [
                        "rll_and_par",
                        "rll_par_mil",
                        "rll_mil_rom",
                    ],
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls[0]["tool_name"] == "get_routes_from_start_to_destination"


def test_navigation_controller_clarifies_vague_destination_description() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_andorra")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-vague-destination",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "I need to change my final destination to a German city known for its cars."
        ),
    )
    assert action is not None
    assert action.action == "respond"
    assert "Which destination" in action.content

    action = controller.decide(
        context_id="ctx-vague-destination",
        messages=messages,
        tools=tools,
        latest_user_text="I meant Munich.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        }
    ]


def test_navigation_controller_deletes_named_intermediate_waypoint() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_mannheim")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-delete-waypoint",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Remove Stuttgart from my route so I can drive directly to Paris "
            "on the shortest route."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        }
    ]

    action = controller.decide(
        context_id="ctx-delete-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": ["loc_mannheim", "loc_stuttgart", "loc_paris"],
                    "routes_to_final_destination_id": [
                        "rll_man_stu",
                        "rll_stu_par",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_mannheim", "name": "Mannheim"},
                            {"id": "loc_stuttgart", "name": "Stuttgart"},
                            {"id": "loc_paris", "name": "Paris"},
                        ]
                    },
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_mannheim",
                "destination_id": "loc_paris",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-delete-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "rll_man_par_fast", "alias": ["fastest"]},
                        {"route_id": "rll_man_par_short", "alias": ["shortest"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_stuttgart",
                "route_id_without_waypoint": "rll_man_par_short",
            },
        }
    ]


def test_navigation_controller_does_not_infer_when_named_waypoint_is_absent() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_andorra")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-delete-absent-waypoint",
        messages=messages,
        tools=tools,
        latest_user_text="Remove Paris from my route and keep the shortest route.",
    )
    action = controller.decide(
        context_id="ctx-delete-absent-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": ["loc_andorra", "loc_milan", "loc_stuttgart"],
                    "routes_to_final_destination_id": [
                        "rll_and_mil",
                        "rll_mil_stu",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_andorra", "name": "Andorra la Vella"},
                            {"id": "loc_milan", "name": "Milan"},
                            {"id": "loc_stuttgart", "name": "Stuttgart"},
                        ]
                    },
                },
            )
        ],
    )

    assert action is not None
    assert action.action == "respond"
    assert "Paris is not currently an intermediate waypoint" in action.content


def test_navigation_controller_defaults_to_fastest_when_deleting_waypoint() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_mannheim")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-delete-fastest-default",
        messages=messages,
        tools=tools,
        latest_user_text="Remove Stuttgart from my route.",
    )
    controller.decide(
        context_id="ctx-delete-fastest-default",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": ["loc_mannheim", "loc_stuttgart", "loc_paris"],
                    "routes_to_final_destination_id": [
                        "rll_man_stu",
                        "rll_stu_par",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_mannheim", "name": "Mannheim"},
                            {"id": "loc_stuttgart", "name": "Stuttgart"},
                            {"id": "loc_paris", "name": "Paris"},
                        ]
                    },
                },
            )
        ],
    )

    action = controller.decide(
        context_id="ctx-delete-fastest-default",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "rll_man_par_fast", "alias": ["fastest"]},
                        {"route_id": "rll_man_par_short", "alias": ["shortest"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_stuttgart",
                "route_id_without_waypoint": "rll_man_par_fast",
            },
        }
    ]


def test_navigation_controller_replaces_single_intermediate_waypoint() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_belgrade")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-replace-waypoint",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Replace my intermediate stop with Frankfurt and use the fastest route."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        }
    ]

    action = controller.decide(
        context_id="ctx-replace-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": ["loc_belgrade", "loc_bucharest", "loc_rome"],
                    "routes_to_final_destination_id": [
                        "rll_bel_buc",
                        "rll_buc_rom",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_belgrade", "name": "Belgrade"},
                            {"id": "loc_bucharest", "name": "Bucharest"},
                            {"id": "loc_rome", "name": "Rome"},
                        ]
                    },
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Frankfurt"},
        }
    ]

    action = controller.decide(
        context_id="ctx-replace-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_frankfurt"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_belgrade",
                "destination_id": "loc_frankfurt",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-replace-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [{"route_id": "rll_bel_fra", "alias": ["fastest"]}]},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_frankfurt",
                "destination_id": "loc_rome",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-replace-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [{"route_id": "rll_fra_rom", "alias": ["fastest"]}]},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_replace_one_waypoint",
            "arguments": {
                "waypoint_id_to_replace": "loc_bucharest",
                "new_waypoint_id": "loc_frankfurt",
                "route_id_leading_to_new_waypoint": "rll_bel_fra",
                "route_id_leading_away_from_new_waypoint": "rll_fra_rom",
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
