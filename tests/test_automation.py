from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from razortrust.audit import AuditLedger
from razortrust.automation import HoldAutomationOrchestrator
from razortrust.domain import HoldCreate
from razortrust.evidence import LegacyEvidenceVerifier
from razortrust.policy import LocalHoldPolicy
from razortrust.repository import InMemoryHoldRepository
from razortrust.risk import CostPolicy, HumanOnlyRiskModel
from razortrust.workflow import HoldService


@pytest.mark.asyncio
async def test_automation_never_fabricates_missing_evaluation_input(tmp_path) -> None:
    repository = InMemoryHoldRepository()
    service = HoldService(
        repository,
        HumanOnlyRiskModel(),
        CostPolicy(),
        LocalHoldPolicy(),
        AuditLedger(tmp_path / "audit.jsonl"),
        LegacyEvidenceVerifier(),
    )
    hold, _ = await service.create_hold(
        HoldCreate(
            request_id=uuid4(),
            merchant_id="merchant_auto",
            source_event_id="event_auto",
            reason_code="AUTOMATION_TEST",
            triggered_at=datetime.now(UTC),
        )
    )
    result = await HoldAutomationOrchestrator(repository, service).run_once()
    assert result.scanned == 1
    assert result.evaluated == 0
    assert result.skipped_missing_input == 1
    assert (await repository.get(hold.hold_id)).state.value == "OPEN"
