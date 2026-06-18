from carbench_agent_core import ToolIndex


def fake_tool(
    name: str,
    *,
    properties: dict | None = None,
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


def test_tool_index_validation_uses_user_safe_missing_capability_template() -> None:
    index = ToolIndex([])

    error = index.validate_call("missing_tool", {})

    assert (
        error
        == "I don't currently have that capability, so I can't complete this request."
    )
    assert "missing_tool" not in error


def test_tool_index_validation_rejects_unknown_arguments_without_leaking_names() -> (
    None
):
    index = ToolIndex(
        [
            fake_tool(
                "set_control",
                properties={"on": {"type": "boolean"}},
            )
        ]
    )

    error = index.validate_call("set_control", {"on": True, "extra": "value"})

    assert error == "I can't complete this request with the available controls."
    assert "extra" not in error


def test_tool_index_validation_rejects_arguments_for_no_arg_tool() -> None:
    index = ToolIndex([fake_tool("get_status")])

    error = index.validate_call("get_status", {"": {}})

    assert error == "I can't complete this request with the available controls."


def test_tool_index_validation_rejects_missing_required_argument() -> None:
    index = ToolIndex(
        [
            fake_tool(
                "set_control",
                properties={"on": {"type": "boolean"}},
                required=["on"],
            )
        ]
    )

    error = index.validate_call("set_control", {})

    assert error == "I need more information before I can complete this request."


def test_tool_index_validation_rejects_enum_and_type_mismatch() -> None:
    index = ToolIndex(
        [
            fake_tool(
                "set_mode",
                properties={
                    "mode": {"type": "string", "enum": ["AUTO", "MANUAL"]},
                    "level": {"type": "integer"},
                },
            )
        ]
    )

    assert (
        index.validate_call("set_mode", {"mode": "SPORT", "level": 1})
        == "I can't complete this request with the available controls."
    )
    assert (
        index.validate_call("set_mode", {"mode": "AUTO", "level": "high"})
        == "I can't complete this request with the available controls."
    )
