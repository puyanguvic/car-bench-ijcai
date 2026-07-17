import pytest
from pydantic import ValidationError

from carbench_agent_core.plan_ir import (
    ActNode,
    AskNode,
    ConfirmNode,
    ObserveNode,
    PlanIR,
    RespondNode,
)


def valid_nodes() -> tuple[object, ...]:
    return (
        ObserveNode(
            id="observe-context",
            operation="read.context",
            success_evidence_key="context.observed",
        ),
        AskNode(
            id="ask-choice",
            prompt="Which option should be used?",
            evidence_key="input.choice",
        ),
        ConfirmNode(
            id="confirm-change",
            depends_on=("observe-context", "ask-choice"),
            prompt="Should I apply the change?",
            evidence_key="input.confirmed",
        ),
        ActNode(
            id="apply-change",
            depends_on=("confirm-change",),
            operation="write.change",
            arguments={"mode": "selected"},
            success_evidence_key="change.applied",
        ),
        RespondNode(
            id="complete",
            depends_on=("apply-change",),
            text="The requested change is complete.",
            completion=True,
            requires_success_evidence=("change.applied",),
        ),
    )


def test_plan_accepts_all_node_kinds_and_returns_stable_dag_order() -> None:
    plan = PlanIR(plan_id="example-plan", nodes=valid_nodes())

    assert plan.topological_order() == (
        "observe-context",
        "ask-choice",
        "confirm-change",
        "apply-change",
        "complete",
    )
    assert set(plan.node_by_id) == {
        "observe-context",
        "ask-choice",
        "confirm-change",
        "apply-change",
        "complete",
    }


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        (
            (
                ObserveNode(
                    id="same",
                    operation="read.value",
                    success_evidence_key="value.one",
                ),
                ActNode(
                    id="same",
                    operation="write.value",
                    success_evidence_key="value.two",
                ),
                RespondNode(
                    id="complete",
                    depends_on=("same",),
                    text="Complete.",
                    completion=True,
                    requires_success_evidence=("value.one",),
                ),
            ),
            "node IDs must be unique",
        ),
        (
            (
                ActNode(
                    id="apply",
                    depends_on=("missing",),
                    operation="write.value",
                    success_evidence_key="value.applied",
                ),
                RespondNode(
                    id="complete",
                    depends_on=("apply",),
                    text="Complete.",
                    completion=True,
                    requires_success_evidence=("value.applied",),
                ),
            ),
            "unknown nodes",
        ),
        (
            (
                ObserveNode(
                    id="first",
                    depends_on=("second",),
                    operation="read.first",
                    success_evidence_key="first.observed",
                ),
                ActNode(
                    id="second",
                    depends_on=("first",),
                    operation="write.second",
                    success_evidence_key="second.applied",
                ),
                RespondNode(
                    id="complete",
                    depends_on=("second",),
                    text="Complete.",
                    completion=True,
                    requires_success_evidence=("second.applied",),
                ),
            ),
            "acyclic graph",
        ),
    ],
)
def test_plan_rejects_invalid_graphs(
    nodes: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        PlanIR(plan_id="invalid-plan", nodes=nodes)


def test_plan_rejects_completion_without_successful_external_evidence() -> None:
    nodes = (
        AskNode(
            id="ask-value",
            prompt="Which value?",
            evidence_key="input.value",
        ),
        RespondNode(
            id="complete",
            depends_on=("ask-value",),
            text="Complete.",
            completion=True,
            requires_success_evidence=("input.value",),
        ),
    )

    with pytest.raises(
        ValidationError,
        match="successful external operation",
    ):
        PlanIR(plan_id="unsafe-completion", nodes=nodes)


def test_plan_rejects_completion_evidence_that_is_not_an_ancestor() -> None:
    nodes = (
        ObserveNode(
            id="orphan-observation",
            operation="read.orphan",
            success_evidence_key="orphan.observed",
        ),
        ActNode(
            id="apply",
            operation="write.value",
            success_evidence_key="value.applied",
        ),
        RespondNode(
            id="complete",
            depends_on=("apply",),
            text="Complete.",
            completion=True,
            requires_success_evidence=("orphan.observed",),
        ),
    )

    with pytest.raises(ValidationError, match="every plan node must lead"):
        PlanIR(plan_id="orphan-evidence", nodes=nodes)


def test_plan_rejects_duplicate_evidence_producers() -> None:
    nodes = (
        ObserveNode(
            id="observe",
            operation="read.value",
            success_evidence_key="shared.evidence",
        ),
        ActNode(
            id="apply",
            depends_on=("observe",),
            operation="write.value",
            success_evidence_key="shared.evidence",
        ),
        RespondNode(
            id="complete",
            depends_on=("apply",),
            text="Complete.",
            completion=True,
            requires_success_evidence=("shared.evidence",),
        ),
    )

    with pytest.raises(ValidationError, match="multiple producers"):
        PlanIR(plan_id="duplicate-evidence", nodes=nodes)

