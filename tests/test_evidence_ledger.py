from carbench_agent_core.evidence_ledger import (
    EventDisposition,
    EvidenceLedger,
    EvidenceRecord,
    EvidenceSource,
    ExternalResultEvent,
    ExternalResultStatus,
)


def external_event(
    *,
    event_id: str = "event-1",
    call_id: str = "call-1",
    payload: object = None,
) -> ExternalResultEvent:
    return ExternalResultEvent(
        event_id=event_id,
        call_id=call_id,
        status=ExternalResultStatus.SUCCESS,
        payload=payload,
    )


def test_ledger_preserves_event_and_call_provenance() -> None:
    ledger = EvidenceLedger()
    event = external_event(payload={"value": 7})

    registration = ledger.register_event(
        event,
        disposition=EventDisposition.APPLIED,
        node_id="observe-value",
    )
    ledger.add(
        EvidenceRecord(
            key="observed.value",
            value=event.payload,
            source=EvidenceSource.EXTERNAL,
            event_id=event.event_id,
            call_id=event.call_id,
            node_id="observe-value",
            operation="read.value",
        )
    )

    assert registration.is_new
    record = ledger.get("observed.value")
    assert record is not None
    assert record.event_id == "event-1"
    assert record.call_id == "call-1"
    assert record.node_id == "observe-value"
    assert record.operation == "read.value"


def test_ledger_reports_identical_replay_as_duplicate() -> None:
    ledger = EvidenceLedger()
    event = external_event(payload={"value": 7})
    ledger.register_event(event, disposition=EventDisposition.APPLIED)

    replay = ledger.register_event(event, disposition=EventDisposition.APPLIED)

    assert not replay.is_new
    assert replay.record.disposition == EventDisposition.DUPLICATE
    assert len(ledger.processed_events) == 1


def test_ledger_rejects_event_id_reuse_with_different_content() -> None:
    ledger = EvidenceLedger()
    ledger.register_event(
        external_event(payload={"value": 7}),
        disposition=EventDisposition.APPLIED,
    )

    conflict = ledger.register_event(
        external_event(payload={"value": 8}),
        disposition=EventDisposition.APPLIED,
    )

    assert not conflict.is_new
    assert conflict.record.disposition == EventDisposition.CONFLICT
    assert len(ledger.processed_events) == 1


def test_ledger_does_not_overwrite_evidence_provenance() -> None:
    ledger = EvidenceLedger()
    first = EvidenceRecord(
        key="observed.value",
        value=7,
        source=EvidenceSource.EXTERNAL,
        event_id="event-1",
        call_id="call-1",
        node_id="observe-value",
    )
    ledger.add(first)
    ledger.add(first)

    different = first.model_copy(update={"event_id": "event-2"})
    try:
        ledger.add(different)
    except ValueError as exc:
        assert "different provenance" in str(exc)
    else:
        raise AssertionError("overwriting evidence should fail")

