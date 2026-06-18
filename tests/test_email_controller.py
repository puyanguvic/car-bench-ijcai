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


def system_messages(*, include_secretary: bool = False) -> list[dict]:
    preference = ""
    if include_secretary:
        preference = (
            '\nUSER_PREFERENCES = {"productivity_and_communication": '
            '{"email": ["If the email is business related, user always '
            "wants to send the mail in addition to original recipient to "
            "secretary: 'emma.walker1132@outlook.com'\"]}}"
        )
    return [
        {
            "role": "system",
            "content": (
                'CURRENT_LOCATION = {"id": "loc_ams_749623", "name": "Amsterdam"}\n'
                'DATETIME = {"year": 2025, "month": 1, "day": 20, '
                '"hour": 14, "minute": 15}'
                f"{preference}"
            ),
        }
    ]


def email_tools(*, include_preferences: bool = False) -> list[dict]:
    tools = [
        fake_tool(
            "get_contact_id_by_contact_name",
            {
                "contact_first_name": {"type": "string"},
                "contact_last_name": {"type": "string"},
            },
        ),
        fake_tool("get_entries_from_calendar"),
        fake_tool(
            "get_contact_information",
            {"contact_ids": {"type": "array", "items": {"type": "string"}}},
            ["contact_ids"],
        ),
        fake_tool(
            "send_email",
            {
                "email_addresses": {"type": "array", "items": {"type": "string"}},
                "content_message": {"type": "string"},
            },
            ["email_addresses", "content_message"],
        ),
    ]
    if include_preferences:
        tools.insert(3, fake_tool("get_user_preferences"))
    return tools


def partnership_calendar() -> dict:
    return {
        "date": {"year": 2025, "month": 1, "day": 20},
        "meetings": [
            {
                "start": {"hour": "14", "minute": "00"},
                "duration": "60min",
                "location": "Minsk",
                "attendees": ["con_5327", "con_8783"],
                "topic": "Partnership Discussion",
            }
        ],
    }


def test_email_controller_confirms_before_sending_late_meeting_email() -> None:
    controller = PolicyAwareController()
    tools = email_tools()
    messages = system_messages(include_secretary=True)

    controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_user_text="What's the status of my 'Partnership Discussion' meeting?",
    )
    controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_entries_from_calendar", partnership_calendar())
        ],
    )

    action = controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Send an email to Frank Walker. I'm running late and apologize "
            "for the delay."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {
                "contact_first_name": "Frank",
                "contact_last_name": "Walker",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_id_by_contact_name",
                {"matches": {"con_1541": "frank walker"}},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": ["con_1541"]},
        }
    ]

    action = controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_information",
                {
                    "con_1541": {
                        "id": "con_1541",
                        "name": {"first_name": "Frank", "last_name": "Walker"},
                        "phone_number": "+49 486 408537",
                        "email": "frank.walker1219@andex.com",
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "confirm" in action.content.lower()
    assert "frank.walker1219@andex.com" in action.content
    assert "emma.walker1132@outlook.com" in action.content
    assert "14:00" in action.content
    assert "15 minutes late" in action.content

    action = controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_user_text="Yes, please send it.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "send_email",
            "arguments": {
                "email_addresses": [
                    "frank.walker1219@andex.com",
                    "emma.walker1132@outlook.com",
                ],
                "content_message": (
                    "Hi Frank, I wanted to reach out regarding our Partnership "
                    "Discussion meeting that started at 14:00 today. I'm running "
                    "about 15 minutes late and apologize for the delay. I should be "
                    "there shortly. Thank you for your patience. Best regards"
                ),
            },
        }
    ]


def test_email_controller_sends_contact_details_to_original_recipient() -> None:
    controller = PolicyAwareController()
    tools = email_tools()
    messages = system_messages()

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Look up Rachel Walker's contact information. I want to send her an email."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {
                "contact_first_name": "Rachel",
                "contact_last_name": "Walker",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_id_by_contact_name",
                {"matches": {"con_3692": "rachel walker"}},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "whose contact details" in action.content.lower()

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_user_text="Share David Harris's contact details.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {
                "contact_first_name": "David",
                "contact_last_name": "Harris",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_id_by_contact_name",
                {"matches": {"con_8528": "david harris"}},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": ["con_3692", "con_8528"]},
        }
    ]

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_information",
                {
                    "con_3692": {
                        "id": "con_3692",
                        "name": {"first_name": "Rachel", "last_name": "Walker"},
                        "phone_number": "+49 913 182721",
                        "email": "rachel.walker1312@outlook.com",
                    },
                    "con_8528": {
                        "id": "con_8528",
                        "name": {"first_name": "David", "last_name": "Harris"},
                        "phone_number": "+49 550 435701",
                        "email": "david.harris3615@protonmail.com",
                    },
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "rachel.walker1312@outlook.com" in action.content
    assert "David Harris" in action.content
    assert "confirm" in action.content.lower()

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_user_text="Yes.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "send_email",
            "arguments": {
                "email_addresses": ["rachel.walker1312@outlook.com"],
                "content_message": (
                    "Hi Rachel,\n\n"
                    "I wanted to share David Harris's contact information with you:\n\n"
                    "Name: David Harris\n"
                    "Phone: +49 550 435701\n"
                    "Email: david.harris3615@protonmail.com\n\n"
                    "Best regards"
                ),
            },
        }
    ]


def test_email_controller_disambiguates_first_name_recipient_before_sending() -> None:
    controller = PolicyAwareController()
    tools = email_tools()
    messages = system_messages()

    action = controller.decide(
        context_id="ctx-share-contact-ambiguous",
        messages=messages,
        tools=tools,
        latest_user_text="Can you help me find Rachel's contact info? I want to send her an email.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {"contact_first_name": "Rachel"},
        }
    ]

    action = controller.decide(
        context_id="ctx-share-contact-ambiguous",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_id_by_contact_name",
                {
                    "matches": {
                        "con_9881": "rachel mitchell",
                        "con_3692": "rachel walker",
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "multiple matching contacts" in action.content.lower()
    assert "Rachel Walker" in action.content

    action = controller.decide(
        context_id="ctx-share-contact-ambiguous",
        messages=messages,
        tools=tools,
        latest_user_text="Rachel Walker.",
    )
    assert action is not None
    assert action.action == "respond"
    assert "whose contact details" in action.content.lower()


def test_email_controller_preserves_preference_recipients_after_confirmation() -> None:
    controller = PolicyAwareController()
    tools = email_tools(include_preferences=True)
    messages = system_messages()

    controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_user_text="What's the status of my 'Partnership Discussion' meeting?",
    )
    controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_entries_from_calendar", partnership_calendar())
        ],
    )
    controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Can you send an email to Frank Walker for me? "
            "I need to let him know I'm running late and apologize."
        ),
    )
    controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_id_by_contact_name",
                {"matches": {"con_1541": "frank walker"}},
            )
        ],
    )
    action = controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_information",
                {
                    "con_1541": {
                        "id": "con_1541",
                        "name": {"first_name": "Frank", "last_name": "Walker"},
                        "phone_number": "+49 486 408537",
                        "email": "frank.walker1219@andex.com",
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_user_preferences",
            "arguments": {
                "preference_categories": {
                    "productivity_and_communication": {"email": True}
                }
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_user_preferences",
                {
                    "productivity_and_communication": {
                        "email": [
                            "If the email is business related, user always wants "
                            "to send the mail in addition to original recipient "
                            "to secretary: 'emma.walker1132@outlook.com'"
                        ]
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "emma.walker1132@outlook.com" in action.content

    action = controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_user_text="Yes.",
    )
    assert action is not None
    assert action.tool_calls[0]["tool_name"] == "send_email"
    assert action.tool_calls[0]["arguments"]["email_addresses"] == [
        "frank.walker1219@andex.com",
        "emma.walker1132@outlook.com",
    ]
