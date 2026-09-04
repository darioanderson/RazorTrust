from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from razortrust.audit import AuditLedger
from razortrust.domain import (
    AnalystReviewSubmission,
    HoldCreate,
    HoldDecision,
    HoldEvaluationInput,
    HumanReviewAction,
    MerchantBaseline,
    TransactionEvent,
    utc_now,
)
from razortrust.policy import LocalHoldPolicy
from razortrust.repository import InMemoryHoldRepository
from razortrust.risk import CostPolicy, DevelopmentRiskModel
from razortrust.workflow import HoldService

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    now = utc_now()
    service = HoldService(
        InMemoryHoldRepository(),
        DevelopmentRiskModel(),
        CostPolicy(),
        LocalHoldPolicy(),
        AuditLedger(ROOT / "var" / "demo-audit-v2.jsonl"),
    )
    hold, _ = await service.create_hold(
        HoldCreate(
            request_id=uuid4(),
            merchant_id="merchant_demo",
            source_event_id="settlement_demo",
            triggered_at=now,
            reason_code="VOLUME_SPIKE",
        )
    )
    baseline = MerchantBaseline(
        volume_mean=4,
        volume_std=1,
        gmv_mean=4000,
        gmv_std=750,
        ticket_size_mean=1000,
        ticket_size_std=200,
        refund_rate_mean=0.02,
        refund_rate_std=0.01,
        chargeback_rate_mean=0.005,
        chargeback_rate_std=0.005,
        known_devices={"device_1", "device_2"},
        known_geos={"IN-MH", "IN-KA"},
        amount_bin_edges=[0, 500, 1000, 2000, 5000],
        amount_bin_probabilities=[0.10, 0.35, 0.45, 0.10],
    )
    transactions = [
        TransactionEvent(
            transaction_id=f"txn_{index}",
            merchant_id=hold.merchant_id,
            timestamp=now - timedelta(hours=4 - index),
            amount=amount,
            device_fingerprint=f"device_{1 + index % 2}",
            customer_geo="IN-MH",
            auth_status="APPROVED",
        )
        for index, amount in enumerate((850, 950, 1100, 1050))
    ]
    decision = await service.evaluate(
        hold.hold_id,
        HoldEvaluationInput(baseline=baseline, transactions=transactions),
    )
    print("AI recommendation (never an authorization):")
    print(decision.model_dump_json(indent=2))
    if decision.decision == HoldDecision.RELEASE:
        review = await service.record_analyst_review(
            hold.hold_id,
            AnalystReviewSubmission(
                request_id=uuid4(),
                action=HumanReviewAction.APPROVE_RELEASE,
                reason_code="DEMO_VERIFIED",
                rationale="The reviewer verified the stable merchant pattern and payment context.",
                decided_at=utc_now(),
                agent_id="razortrust-demo-agent",
                agent_session="demo-session",
                delegated_permissions=["REVIEW_SETTLEMENT"],
                authorized_amount=sum(transaction.amount for transaction in transactions),
                authorized_item=hold.source_event_id,
                infrastructure_provider="local-demo",
                transaction_identity=hold.source_event_id,
            ),
            analyst_id="demo-analyst",
        )
        print("Human authorization:")
        print(review.model_dump_json(indent=2))
        print(f"final_hold_state={(await service.repository.get(hold.hold_id)).state}")
    valid, record_count = service.ledger.verify()
    print(f"audit_valid={valid} records={record_count}")


if __name__ == "__main__":
    asyncio.run(main())
