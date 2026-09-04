from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from razortrust.checkout_bridge import (
    CheckoutOrderRow,
    CheckoutPaymentAttemptRow,
    SqlCheckoutBridgeStore,
)
from razortrust.live_features import SOURCE_CONTRACT_VERSION, FirstPartyTelemetryRow


@pytest.mark.asyncio
async def test_failed_then_captured_attempts_are_preserved_without_overwriting_telemetry() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(CheckoutOrderRow.__table__.create)
        await conn.run_sync(CheckoutPaymentAttemptRow.__table__.create)
        await conn.run_sync(FirstPartyTelemetryRow.__table__.create)

    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlCheckoutBridgeStore(sessions)
    checkout_session_id = uuid4()
    telemetry_id = uuid4()
    now = datetime(2026, 9, 2, 9, 31, tzinfo=UTC)

    async with sessions() as session:
        session.add(
            FirstPartyTelemetryRow(
                telemetry_id=telemetry_id,
                account_id="acc_test",
                order_id="order_multi_attempt_1",
                payment_id=None,
                device_fingerprint_sha256="a" * 64,
                customer_geo="IN-UP",
                client_event_at=now,
                observed_at=now,
                payload_sha256="b" * 64,
                source_contract_version=SOURCE_CONTRACT_VERSION,
            )
        )
        session.add(
            CheckoutOrderRow(
                checkout_session_id=checkout_session_id,
                account_id="acc_test",
                razorpay_order_id="order_multi_attempt_1",
                razorpay_payment_id=None,
                amount=1000,
                currency="INR",
                receipt="rt-multi-attempt-1",
                provider_status="created",
                provider_attempts=0,
                lifecycle_status="OPEN",
                last_payment_status=None,
                device_fingerprint_sha256="a" * 64,
                customer_geo="IN-UP",
                client_event_at=now,
                telemetry_id=telemetry_id,
                authoritative_payment_status=None,
                signature_verified_at=None,
                webhook_confirmed_at=None,
                last_provider_reconciled_at=None,
                expires_at=None,
                abandoned_at=None,
                created_at=now,
                updated_at=now,
                checkout_contract_version="razorpay-standard-checkout-bridge@1",
            )
        )
        await session.commit()

    await store.observe_payment_event(
        event_type="payment.failed",
        source_event_id="evt_failed_1",
        order_id="order_multi_attempt_1",
        payment_id="pay_failed_1",
        payment_status="failed",
        captured=False,
        observed_at=now,
    )

    async with sessions() as session:
        checkout = await session.get(CheckoutOrderRow, checkout_session_id)
        telemetry = await session.get(FirstPartyTelemetryRow, telemetry_id)
        attempts = list((await session.scalars(select(CheckoutPaymentAttemptRow))).all())
        assert checkout is not None
        assert telemetry is not None
        assert checkout.lifecycle_status == "ATTEMPT_FAILED"
        assert checkout.provider_attempts == 1
        assert checkout.razorpay_payment_id is None
        assert telemetry.payment_id is None
        assert [(a.razorpay_payment_id, a.payment_status) for a in attempts] == [
            ("pay_failed_1", "failed")
        ]

    later = datetime(2026, 9, 2, 9, 32, tzinfo=UTC)
    await store.observe_payment_event(
        event_type="payment.captured",
        source_event_id="evt_captured_2",
        order_id="order_multi_attempt_1",
        payment_id="pay_captured_2",
        payment_status="captured",
        captured=True,
        observed_at=later,
    )

    async with sessions() as session:
        checkout = await session.get(CheckoutOrderRow, checkout_session_id)
        telemetry = await session.get(FirstPartyTelemetryRow, telemetry_id)
        attempts = list(
            (
                await session.scalars(
                    select(CheckoutPaymentAttemptRow).order_by(
                        CheckoutPaymentAttemptRow.first_seen_at,
                        CheckoutPaymentAttemptRow.razorpay_payment_id,
                    )
                )
            ).all()
        )
        assert checkout is not None
        assert telemetry is not None
        assert checkout.lifecycle_status == "PAID"
        assert checkout.provider_attempts == 2
        assert checkout.razorpay_payment_id == "pay_captured_2"
        assert telemetry.payment_id == "pay_captured_2"
        assert [(a.razorpay_payment_id, a.payment_status) for a in attempts] == [
            ("pay_failed_1", "failed"),
            ("pay_captured_2", "captured"),
        ]

    await engine.dispose()
