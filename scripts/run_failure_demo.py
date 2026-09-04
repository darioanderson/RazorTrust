from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from razortrust.audit import AuditLedger
from razortrust.domain import (
    HoldCreate,
    HoldEvaluationInput,
    MerchantBaseline,
    TransactionEvent,
    utc_now,
)
from razortrust.policy import LocalHoldPolicy
from razortrust.repository import InMemoryHoldRepository
from razortrust.risk import CostPolicy
from razortrust.workflow import HoldService

ROOT = Path(__file__).resolve().parents[1]


class UnavailableModel:
    version = "unavailable-demo-model@1"

    def predict(self, _features, cohort="pooled"):
        raise RuntimeError("simulated model artifact outage")


async def main() -> None:
    now = utc_now()
    service = HoldService(
        InMemoryHoldRepository(),
        UnavailableModel(),
        CostPolicy(),
        LocalHoldPolicy(),
        AuditLedger(ROOT / "var" / "failure-demo-audit.jsonl"),
    )
    hold, _ = await service.create_hold(
        HoldCreate(
            request_id=uuid4(),
            merchant_id="merchant_failure_demo",
            source_event_id="settlement_failure_demo",
            triggered_at=now,
            reason_code="MODEL_OUTAGE",
        )
    )
    evaluation = HoldEvaluationInput(
        baseline=MerchantBaseline(
            volume_mean=2,
            volume_std=1,
            gmv_mean=2000,
            gmv_std=500,
            ticket_size_mean=1000,
            ticket_size_std=200,
            refund_rate_mean=0.01,
            refund_rate_std=0.01,
            chargeback_rate_mean=0.005,
            chargeback_rate_std=0.005,
            known_devices={"device_known"},
            known_geos={"IN-MH"},
            amount_bin_edges=[0, 500, 1000, 2000],
            amount_bin_probabilities=[0.1, 0.4, 0.5],
        ),
        transactions=[
            TransactionEvent(
                transaction_id="txn_failure_demo",
                merchant_id=hold.merchant_id,
                timestamp=now - timedelta(minutes=15),
                amount=950,
                device_fingerprint="device_known",
                customer_geo="IN-MH",
                auth_status="APPROVED",
            )
        ],
    )
    decision = await service.evaluate(hold.hold_id, evaluation)
    final_hold = await service.repository.get(hold.hold_id)
    valid, count = service.ledger.verify()
    print("simulated_failure=model_unavailable")
    print(f"decision={decision.decision}")
    print(f"reason={decision.reason_code}")
    print(f"hold_state={final_hold.state}")
    print(f"audit_valid={valid} records={count}")


if __name__ == "__main__":
    asyncio.run(main())
