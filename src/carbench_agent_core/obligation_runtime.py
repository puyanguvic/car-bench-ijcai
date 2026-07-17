"""Deterministic runtime for validated obligation-plan DAGs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .evidence_ledger import (
    ConfirmationEvent,
    EventDisposition,
    EvidenceLedger,
    EvidenceRecord,
    EvidenceSource,
    ExternalResultEvent,
    ExternalResultStatus,
    InputEvent,
    RuntimeEvent,
)
from .plan_ir import (
    ActNode,
    AskNode,
    ConfirmNode,
    ObserveNode,
    PlanIR,
    PlanNode,
    RespondNode,
)


class NodeExecutionStatus(StrEnum):
    PENDING = "pending"
    WAITING = "waiting"
    SATISFIED = "satisfied"
    FAILED = "failed"


class PlanRunStatus(StrEnum):
    ACTIVE = "active"
    WAITING = "waiting"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class ExternalCallAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["observe", "act"]
    node_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class AskAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ask"] = "ask"
    node_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class ConfirmAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["confirm"] = "confirm"
    node_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class RespondAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["respond"] = "respond"
    node_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    completion: bool
    evidence_refs: tuple[str, ...] = ()


RuntimeAction: TypeAlias = (
    ExternalCallAction | AskAction | ConfirmAction | RespondAction
)


@dataclass(frozen=True)
class PendingAction:
    """The single emitted action currently awaiting a correlated event."""

    node_id: str
    call_id: str
    kind: Literal["observe", "ask", "confirm", "act"]
    operation: str | None = None


@dataclass(frozen=True)
class EventApplication:
    """Public audit result of applying one runtime event."""

    disposition: EventDisposition
    event_id: str
    call_id: str
    node_id: str | None


CallIdFactory: TypeAlias = Callable[[], str]


@dataclass
class ObligationRuntime:
    """Execute one immutable plan with exactly-once event semantics."""

    plan: PlanIR
    call_id_factory: CallIdFactory = field(
        default=lambda: f"call_{uuid4().hex}", repr=False
    )
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    _node_status: dict[str, NodeExecutionStatus] = field(init=False, repr=False)
    _pending: PendingAction | None = field(default=None, init=False, repr=False)
    _issued_call_ids: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._node_status = {
            node.id: NodeExecutionStatus.PENDING for node in self.plan.nodes
        }

    @property
    def pending_action(self) -> PendingAction | None:
        return self._pending

    @property
    def status(self) -> PlanRunStatus:
        statuses = tuple(self._node_status.values())
        if any(status == NodeExecutionStatus.FAILED for status in statuses):
            return PlanRunStatus.BLOCKED
        if all(status == NodeExecutionStatus.SATISFIED for status in statuses):
            return PlanRunStatus.COMPLETED
        if self._pending is not None:
            return PlanRunStatus.WAITING
        return PlanRunStatus.ACTIVE

    def node_status(self, node_id: str) -> NodeExecutionStatus:
        return self._node_status[node_id]

    def step(self) -> RuntimeAction | None:
        """Emit the next ready action once, or return ``None`` while not runnable."""

        if self.status in {
            PlanRunStatus.WAITING,
            PlanRunStatus.BLOCKED,
            PlanRunStatus.COMPLETED,
        }:
            return None

        for node_id in self.plan.topological_order():
            if self._node_status[node_id] != NodeExecutionStatus.PENDING:
                continue
            node = self.plan.node_by_id[node_id]
            if not self._dependencies_satisfied(node):
                continue
            return self._emit(node)
        return None

    def apply_event(self, event: RuntimeEvent) -> EventApplication:
        """Apply a correlated event without replaying or reordering effects."""

        pending = self._pending
        preliminary = self.ledger.register_event(
            event,
            disposition=(
                EventDisposition.APPLIED
                if pending is not None and event.call_id == pending.call_id
                else EventDisposition.OUT_OF_ORDER
            ),
            node_id=(
                pending.node_id
                if pending is not None and event.call_id == pending.call_id
                else None
            ),
        )
        if not preliminary.is_new:
            return EventApplication(
                disposition=preliminary.record.disposition,
                event_id=event.event_id,
                call_id=event.call_id,
                node_id=preliminary.record.node_id,
            )

        if pending is None or event.call_id != pending.call_id:
            return EventApplication(
                disposition=EventDisposition.OUT_OF_ORDER,
                event_id=event.event_id,
                call_id=event.call_id,
                node_id=None,
            )

        node = self.plan.node_by_id[pending.node_id]
        if not _event_matches_node(event, node):
            # The event is consumed as out of order, but the legitimate pending
            # action remains available for a correctly typed event.
            self._replace_event_disposition(event, EventDisposition.OUT_OF_ORDER)
            return EventApplication(
                disposition=EventDisposition.OUT_OF_ORDER,
                event_id=event.event_id,
                call_id=event.call_id,
                node_id=pending.node_id,
            )

        if isinstance(event, ExternalResultEvent):
            if event.status == ExternalResultStatus.FAILURE:
                self._node_status[pending.node_id] = NodeExecutionStatus.FAILED
                self._pending = None
                self._replace_event_disposition(event, EventDisposition.FAILURE)
                return EventApplication(
                    disposition=EventDisposition.FAILURE,
                    event_id=event.event_id,
                    call_id=event.call_id,
                    node_id=pending.node_id,
                )
            evidence_key = _evidence_key(node)
            assert evidence_key is not None
            self.ledger.add(
                EvidenceRecord(
                    key=evidence_key,
                    value=event.payload,
                    source=EvidenceSource.EXTERNAL,
                    event_id=event.event_id,
                    call_id=event.call_id,
                    node_id=pending.node_id,
                    operation=pending.operation,
                )
            )
        elif isinstance(event, InputEvent):
            assert isinstance(node, AskNode)
            self.ledger.add(
                EvidenceRecord(
                    key=node.evidence_key,
                    value=event.value,
                    source=EvidenceSource.INPUT,
                    event_id=event.event_id,
                    call_id=event.call_id,
                    node_id=pending.node_id,
                )
            )
        else:
            assert isinstance(event, ConfirmationEvent)
            assert isinstance(node, ConfirmNode)
            self.ledger.add(
                EvidenceRecord(
                    key=node.evidence_key,
                    value=event.confirmed,
                    source=EvidenceSource.INPUT,
                    event_id=event.event_id,
                    call_id=event.call_id,
                    node_id=pending.node_id,
                )
            )
            if not event.confirmed:
                self._node_status[pending.node_id] = NodeExecutionStatus.FAILED
                self._pending = None
                self._replace_event_disposition(event, EventDisposition.FAILURE)
                return EventApplication(
                    disposition=EventDisposition.FAILURE,
                    event_id=event.event_id,
                    call_id=event.call_id,
                    node_id=pending.node_id,
                )

        self._node_status[pending.node_id] = NodeExecutionStatus.SATISFIED
        self._pending = None
        return EventApplication(
            disposition=EventDisposition.APPLIED,
            event_id=event.event_id,
            call_id=event.call_id,
            node_id=pending.node_id,
        )

    def _dependencies_satisfied(self, node: PlanNode) -> bool:
        return all(
            self._node_status[dependency] == NodeExecutionStatus.SATISFIED
            for dependency in node.depends_on
        )

    def _emit(self, node: PlanNode) -> RuntimeAction:
        if isinstance(node, RespondNode):
            evidence_ref_list: list[str] = []
            for key in node.requires_success_evidence:
                evidence = self.ledger.get(key)
                if evidence is not None:
                    evidence_ref_list.append(evidence.event_id)
            evidence_refs = tuple(evidence_ref_list)
            if node.completion and len(evidence_refs) != len(
                node.requires_success_evidence
            ):
                raise RuntimeError(
                    "completion response is missing successful evidence records"
                )
            self._node_status[node.id] = NodeExecutionStatus.SATISFIED
            return RespondAction(
                node_id=node.id,
                text=node.text,
                completion=node.completion,
                evidence_refs=evidence_refs,
            )

        call_id = self.call_id_factory()
        if not call_id:
            raise ValueError("call_id_factory must return a non-empty call ID")
        if call_id in self._issued_call_ids:
            raise ValueError(f"call_id_factory reused call ID {call_id!r}")
        self._issued_call_ids.add(call_id)
        if isinstance(node, ObserveNode):
            action: RuntimeAction = ExternalCallAction(
                kind="observe",
                node_id=node.id,
                call_id=call_id,
                operation=node.operation,
                arguments=node.arguments,
            )
            operation = node.operation
        elif isinstance(node, ActNode):
            action = ExternalCallAction(
                kind="act",
                node_id=node.id,
                call_id=call_id,
                operation=node.operation,
                arguments=node.arguments,
            )
            operation = node.operation
        elif isinstance(node, AskNode):
            action = AskAction(
                node_id=node.id,
                call_id=call_id,
                prompt=node.prompt,
            )
            operation = None
        else:
            assert isinstance(node, ConfirmNode)
            action = ConfirmAction(
                node_id=node.id,
                call_id=call_id,
                prompt=node.prompt,
            )
            operation = None

        self._node_status[node.id] = NodeExecutionStatus.WAITING
        self._pending = PendingAction(
            node_id=node.id,
            call_id=call_id,
            kind=node.kind,
            operation=operation,
        )
        return action

    def _replace_event_disposition(
        self, event: RuntimeEvent, disposition: EventDisposition
    ) -> None:
        """Update the just-created audit record without exposing mutable models."""

        self.ledger.update_event_disposition(event.event_id, disposition)


def _event_matches_node(event: RuntimeEvent, node: PlanNode) -> bool:
    if isinstance(event, ExternalResultEvent):
        return isinstance(node, (ObserveNode, ActNode))
    if isinstance(event, InputEvent):
        return isinstance(node, AskNode)
    return isinstance(node, ConfirmNode)


def _evidence_key(node: PlanNode) -> str | None:
    if isinstance(node, (ObserveNode, ActNode)):
        return node.success_evidence_key
    if isinstance(node, (AskNode, ConfirmNode)):
        return node.evidence_key
    return None
