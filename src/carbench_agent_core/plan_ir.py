"""Typed, domain-agnostic intermediate representation for obligation plans."""

from __future__ import annotations

from collections import deque
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _NodeBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    depends_on: tuple[str, ...] = ()


class ObserveNode(_NodeBase):
    """Acquire a fact through a read-only external operation."""

    kind: Literal["observe"] = "observe"
    operation: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    success_evidence_key: str = Field(min_length=1)


class AskNode(_NodeBase):
    """Ask for a missing value and record the correlated input."""

    kind: Literal["ask"] = "ask"
    prompt: str = Field(min_length=1)
    evidence_key: str = Field(min_length=1)


class ConfirmNode(_NodeBase):
    """Request an explicit boolean confirmation."""

    kind: Literal["confirm"] = "confirm"
    prompt: str = Field(min_length=1)
    evidence_key: str = Field(min_length=1)


class ActNode(_NodeBase):
    """Execute a state-changing external operation."""

    kind: Literal["act"] = "act"
    operation: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    success_evidence_key: str = Field(min_length=1)


class RespondNode(_NodeBase):
    """Emit text; completion responses require successful external evidence."""

    kind: Literal["respond"] = "respond"
    text: str = Field(min_length=1)
    completion: bool = False
    requires_success_evidence: tuple[str, ...] = ()


PlanNode: TypeAlias = Annotated[
    ObserveNode | AskNode | ConfirmNode | ActNode | RespondNode,
    Field(discriminator="kind"),
]


class PlanIR(BaseModel):
    """Validated DAG whose unique terminal node is a completion response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1)
    nodes: tuple[PlanNode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> "PlanIR":
        node_by_id = {node.id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes):
            raise ValueError("plan node IDs must be unique")

        for node in self.nodes:
            if len(set(node.depends_on)) != len(node.depends_on):
                raise ValueError(f"node {node.id!r} has duplicate dependencies")
            missing = set(node.depends_on) - set(node_by_id)
            if missing:
                raise ValueError(
                    f"node {node.id!r} depends on unknown nodes: {sorted(missing)}"
                )
            if node.id in node.depends_on:
                raise ValueError(f"node {node.id!r} cannot depend on itself")

        order = _topological_order(self.nodes)
        if len(order) != len(self.nodes):
            raise ValueError("plan dependencies must form an acyclic graph")

        completions = [
            node
            for node in self.nodes
            if isinstance(node, RespondNode) and node.completion
        ]
        if len(completions) != 1:
            raise ValueError("plan must contain exactly one completion response")
        completion = completions[0]

        dependents = _dependents(self.nodes)
        if dependents[completion.id]:
            raise ValueError("the completion response must be terminal")

        ancestors = _ancestors(completion.id, node_by_id)
        unreachable = set(node_by_id) - ancestors - {completion.id}
        if unreachable:
            raise ValueError(
                "every plan node must lead to the completion response; "
                f"unreachable nodes: {sorted(unreachable)}"
            )

        if not completion.requires_success_evidence:
            raise ValueError("completion must require successful external evidence")
        if len(set(completion.requires_success_evidence)) != len(
            completion.requires_success_evidence
        ):
            raise ValueError("completion evidence requirements must be unique")

        producer_by_key: dict[str, str] = {}
        for node in self.nodes:
            evidence_key = _produced_evidence_key(node)
            if evidence_key is None:
                continue
            if evidence_key in producer_by_key:
                raise ValueError(
                    f"evidence key {evidence_key!r} has multiple producers"
                )
            producer_by_key[evidence_key] = node.id

        for key in completion.requires_success_evidence:
            producer_id = producer_by_key.get(key)
            if producer_id is None:
                raise ValueError(
                    f"completion requires evidence {key!r} with no producer"
                )
            producer = node_by_id[producer_id]
            if not isinstance(producer, (ObserveNode, ActNode)):
                raise ValueError(
                    "completion evidence must come from a successful external operation"
                )
            if producer_id not in ancestors:
                raise ValueError(
                    f"evidence producer {producer_id!r} is not an ancestor of completion"
                )

        return self

    @property
    def node_by_id(self) -> dict[str, PlanNode]:
        return {node.id: node for node in self.nodes}

    def topological_order(self) -> tuple[str, ...]:
        return _topological_order(self.nodes)


def _produced_evidence_key(node: PlanNode) -> str | None:
    if isinstance(node, (ObserveNode, ActNode)):
        return node.success_evidence_key
    if isinstance(node, (AskNode, ConfirmNode)):
        return node.evidence_key
    return None


def _dependents(nodes: tuple[PlanNode, ...]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {node.id: set() for node in nodes}
    for node in nodes:
        for dependency in node.depends_on:
            result[dependency].add(node.id)
    return result


def _topological_order(nodes: tuple[PlanNode, ...]) -> tuple[str, ...]:
    """Stable Kahn ordering, preserving declaration order among ready nodes."""

    by_id = {node.id: node for node in nodes}
    position = {node.id: index for index, node in enumerate(nodes)}
    indegree = {node.id: len(node.depends_on) for node in nodes}
    dependents = _dependents(nodes)
    ready = deque(node.id for node in nodes if indegree[node.id] == 0)
    ordered: list[str] = []

    while ready:
        node_id = ready.popleft()
        ordered.append(node_id)
        newly_ready: list[str] = []
        for dependent in dependents[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                newly_ready.append(dependent)
        for dependent in sorted(newly_ready, key=position.__getitem__):
            ready.append(dependent)

    return tuple(node_id for node_id in ordered if node_id in by_id)


def _ancestors(node_id: str, node_by_id: dict[str, PlanNode]) -> set[str]:
    result: set[str] = set()
    frontier = list(node_by_id[node_id].depends_on)
    while frontier:
        ancestor = frontier.pop()
        if ancestor in result:
            continue
        result.add(ancestor)
        frontier.extend(node_by_id[ancestor].depends_on)
    return result
