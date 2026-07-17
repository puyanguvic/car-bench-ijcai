from collections.abc import Iterator

from carbench_agent_core.evidence_ledger import (
    ConfirmationEvent,
    EventDisposition,
    ExternalResultEvent,
    ExternalResultStatus,
    InputEvent,
)
from carbench_agent_core.obligation_runtime import (
    AskAction,
    ConfirmAction,
    ExternalCallAction,
    NodeExecutionStatus,
    ObligationRuntime,
    PlanRunStatus,
    RespondAction,
)
from carbench_agent_core.plan_ir import (
    ActNode,
    AskNode,
    ConfirmNode,
    ObserveNode,
    PlanIR,
    RespondNode,
)


def call_ids(*values: str):
    iterator: Iterator[str] = iter(values)
    return iterator.__next__


def full_plan() -> PlanIR:
    return PlanIR(
        plan_id="runtime-example",
        nodes=(
            ObserveNode(
                id="observe-context",
                operation="read.context",
                success_evidence_key="context.observed",
            ),
            AskNode(
                id="ask-choice",
                depends_on=("observe-context",),
                prompt="Which option should be used?",
                evidence_key="input.choice",
            ),
            ConfirmNode(
                id="confirm-change",
                depends_on=("ask-choice",),
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
        ),
    )


def test_runtime_executes_all_node_kinds_with_provenance() -> None:
    runtime = ObligationRuntime(
        full_plan(),
        call_id_factory=call_ids("call-observe", "call-ask", "call-confirm", "call-act"),
    )

    observe = runtime.step()
    assert observe == ExternalCallAction(
        kind="observe",
        node_id="observe-context",
        call_id="call-observe",
        operation="read.context",
        arguments={},
    )
    assert runtime.step() is None
    assert runtime.status == PlanRunStatus.WAITING

    observed = runtime.apply_event(
        ExternalResultEvent(
            event_id="event-observe",
            call_id="call-observe",
            status=ExternalResultStatus.SUCCESS,
            payload={"state": "ready"},
        )
    )
    assert observed.disposition == EventDisposition.APPLIED

    ask = runtime.step()
    assert isinstance(ask, AskAction)
    assert ask.call_id == "call-ask"
    runtime.apply_event(
        InputEvent(
            event_id="event-input",
            call_id="call-ask",
            value="option-a",
        )
    )

    confirm = runtime.step()
    assert isinstance(confirm, ConfirmAction)
    assert confirm.call_id == "call-confirm"
    runtime.apply_event(
        ConfirmationEvent(
            event_id="event-confirm",
            call_id="call-confirm",
            confirmed=True,
        )
    )

    act = runtime.step()
    assert act == ExternalCallAction(
        kind="act",
        node_id="apply-change",
        call_id="call-act",
        operation="write.change",
        arguments={"mode": "selected"},
    )
    runtime.apply_event(
        ExternalResultEvent(
            event_id="event-act",
            call_id="call-act",
            status=ExternalResultStatus.SUCCESS,
            payload={"applied": True},
        )
    )

    response = runtime.step()
    assert response == RespondAction(
        node_id="complete",
        text="The requested change is complete.",
        completion=True,
        evidence_refs=("event-act",),
    )
    assert runtime.status == PlanRunStatus.COMPLETED
    assert runtime.step() is None

    evidence = runtime.ledger.get("change.applied")
    assert evidence is not None
    assert evidence.event_id == "event-act"
    assert evidence.call_id == "call-act"
    assert evidence.node_id == "apply-change"
    assert evidence.operation == "write.change"


def test_failure_never_satisfies_action_or_emits_completion() -> None:
    plan = PlanIR(
        plan_id="failure-example",
        nodes=(
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
                requires_success_evidence=("value.applied",),
            ),
        ),
    )
    runtime = ObligationRuntime(plan, call_id_factory=call_ids("call-act"))
    action = runtime.step()
    assert isinstance(action, ExternalCallAction)

    event = ExternalResultEvent(
        event_id="event-failure",
        call_id="call-act",
        status=ExternalResultStatus.FAILURE,
        error={"code": "rejected"},
    )
    result = runtime.apply_event(event)

    assert result.disposition == EventDisposition.FAILURE
    assert runtime.status == PlanRunStatus.BLOCKED
    assert runtime.node_status("apply") == NodeExecutionStatus.FAILED
    assert not runtime.ledger.has("value.applied")
    assert runtime.step() is None

    replay = runtime.apply_event(event)
    assert replay.disposition == EventDisposition.DUPLICATE
    assert runtime.status == PlanRunStatus.BLOCKED


def test_out_of_order_and_wrong_call_results_are_idempotently_ignored() -> None:
    plan = PlanIR(
        plan_id="ordering-example",
        nodes=(
            ObserveNode(
                id="observe",
                operation="read.value",
                success_evidence_key="value.observed",
            ),
            RespondNode(
                id="complete",
                depends_on=("observe",),
                text="Complete.",
                completion=True,
                requires_success_evidence=("value.observed",),
            ),
        ),
    )
    runtime = ObligationRuntime(plan, call_id_factory=call_ids("call-observe"))

    early = ExternalResultEvent(
        event_id="event-early",
        call_id="call-observe",
        status=ExternalResultStatus.SUCCESS,
        payload={"value": 3},
    )
    assert runtime.apply_event(early).disposition == EventDisposition.OUT_OF_ORDER
    assert not runtime.ledger.has("value.observed")

    observe = runtime.step()
    assert isinstance(observe, ExternalCallAction)
    assert runtime.apply_event(early).disposition == EventDisposition.DUPLICATE
    assert runtime.status == PlanRunStatus.WAITING

    wrong_call = ExternalResultEvent(
        event_id="event-wrong-call",
        call_id="call-other",
        status=ExternalResultStatus.SUCCESS,
        payload={"value": 4},
    )
    assert (
        runtime.apply_event(wrong_call).disposition
        == EventDisposition.OUT_OF_ORDER
    )
    assert runtime.status == PlanRunStatus.WAITING

    correct = ExternalResultEvent(
        event_id="event-correct",
        call_id="call-observe",
        status=ExternalResultStatus.SUCCESS,
        payload={"value": 5},
    )
    assert runtime.apply_event(correct).disposition == EventDisposition.APPLIED
    assert runtime.ledger.get("value.observed").value == {"value": 5}
    assert isinstance(runtime.step(), RespondAction)


def test_same_call_with_wrong_event_type_does_not_consume_pending_action() -> None:
    plan = PlanIR(
        plan_id="type-correlation-example",
        nodes=(
            AskNode(
                id="ask",
                prompt="Which value?",
                evidence_key="input.value",
            ),
            ActNode(
                id="apply",
                depends_on=("ask",),
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
    )
    runtime = ObligationRuntime(
        plan,
        call_id_factory=call_ids("call-ask", "call-act"),
    )
    assert isinstance(runtime.step(), AskAction)

    wrong_type = ExternalResultEvent(
        event_id="event-wrong-type",
        call_id="call-ask",
        status=ExternalResultStatus.SUCCESS,
        payload="value-a",
    )
    assert (
        runtime.apply_event(wrong_type).disposition
        == EventDisposition.OUT_OF_ORDER
    )
    assert runtime.pending_action is not None
    assert runtime.pending_action.call_id == "call-ask"

    accepted = runtime.apply_event(
        InputEvent(
            event_id="event-input",
            call_id="call-ask",
            value="value-a",
        )
    )
    assert accepted.disposition == EventDisposition.APPLIED
    assert isinstance(runtime.step(), ExternalCallAction)


def test_declined_confirmation_blocks_descendants() -> None:
    plan = PlanIR(
        plan_id="confirmation-example",
        nodes=(
            ConfirmNode(
                id="confirm",
                prompt="Should I continue?",
                evidence_key="input.confirmed",
            ),
            ActNode(
                id="apply",
                depends_on=("confirm",),
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
    )
    runtime = ObligationRuntime(plan, call_id_factory=call_ids("call-confirm"))
    assert isinstance(runtime.step(), ConfirmAction)

    rejected = runtime.apply_event(
        ConfirmationEvent(
            event_id="event-decline",
            call_id="call-confirm",
            confirmed=False,
        )
    )

    assert rejected.disposition == EventDisposition.FAILURE
    assert runtime.status == PlanRunStatus.BLOCKED
    assert runtime.node_status("apply") == NodeExecutionStatus.PENDING
    assert runtime.step() is None


def test_runtime_rejects_reused_call_ids() -> None:
    runtime = ObligationRuntime(
        full_plan(),
        call_id_factory=lambda: "call-reused",
    )
    assert isinstance(runtime.step(), ExternalCallAction)
    runtime.apply_event(
        ExternalResultEvent(
            event_id="event-observe",
            call_id="call-reused",
            status=ExternalResultStatus.SUCCESS,
            payload={"state": "ready"},
        )
    )

    try:
        runtime.step()
    except ValueError as exc:
        assert "reused call ID" in str(exc)
    else:
        raise AssertionError("reusing a call ID should fail")
