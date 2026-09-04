from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.responses import Response as FastAPIResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .alerts import CriticalFailureMonitor, LoggingAlertSink
from .audit import AuditFormatError, AuditLedger
from .authorization import (
    AuthorizationUnavailable,
    LocalAuthorizer,
    OpaAuthorizer,
)
from .automation import HoldAutomationOrchestrator
from .checkout_bridge import (
    CheckoutOrderCreate,
    CheckoutOrderResponse,
    CheckoutVerificationResponse,
    CheckoutVerificationSubmission,
    SqlCheckoutBridgeStore,
    validate_authoritative_payment,
    verify_checkout_signature,
)
from .database import SqlHoldRepository
from .domain import (
    ActorRole,
    AnalystReviewRecord,
    AnalystReviewSubmission,
    AuthorizationRequest,
    DecisionResponse,
    EvidenceRecord,
    EvidenceSubmission,
    HoldCase,
    HoldCreate,
    HoldDecision,
    HoldEvaluationInput,
    PolicyCheck,
    PolicyResult,
    Principal,
)
from .dossier import build_evidence_dossier
from .evidence import (
    LegacyEvidenceVerifier,
    SignedAttestationEvidenceVerifier,
    UnavailableEvidenceVerifier,
)
from .ground_truth import SqlProviderGroundTruthStore
from .integrations.razorpay import (
    InMemoryRazorpayEventStore,
    RazorpayApiError,
    RazorpayClient,
    RazorpayWebhookEnvelope,
    build_event_summary,
    payload_sha256,
    verify_webhook_signature,
)
from .layer_execution import LayerExecutionService
from .live_features import FirstPartyTelemetrySubmission, SqlLiveFeatureStore
from .model_governance import model_governance_status
from .observability import (
    bind_request_trace_id,
    configure_observability,
    reset_request_trace_id,
)
from .operator_history import (
    HistoryImportValidationError,
    OperatorHistoryStore,
    audit_csv_bytes,
)
from .payment_case_bridge import (
    PaymentCaseBridgeUnavailable,
    RazorpayPaymentCaseBridge,
    SqlRazorpayPaymentCaseReader,
)
from .policy import LocalHoldPolicy, OpaHoldPolicy, PolicyUnavailable
from .processor_timeline import SqlPointInTimeTimelineStore
from .public_benchmark import PublicBenchmarkService
from .razorpay_processor import (
    InMemoryRazorpayReconstructionStore,
    RazorpayPaymentReconstructor,
)
from .razorpay_store import SqlRazorpayEventStore, SqlRazorpayReconstructionStore
from .repository import (
    HoldNotFound,
    HoldRepository,
    InMemoryHoldRepository,
    InvalidTransition,
    TransactionalAuditRepository,
)
from .research_status import FINAL_V3_RESEARCH_STATUS, PHASE_IMPLEMENTATION_STATUS
from .risk import (
    CostPolicy,
    DevelopmentRiskModel,
    HumanOnlyRiskModel,
    RiskModel,
    UnavailableRiskModel,
)
from .security_middleware import ProductionSecurityMiddleware
from .settings import Settings
from .shadow_runtime import load_shadow_analysis_runtime
from .shadow_scoring import ShadowScoringService, SqlShadowScoreStore
from .workflow import HoldService


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    failure_monitor = CriticalFailureMonitor(LoggingAlertSink())
    engine = None
    repository: HoldRepository
    razorpay_event_store: SqlRazorpayEventStore | InMemoryRazorpayEventStore
    razorpay_reconstruction_store: (
        SqlRazorpayReconstructionStore | InMemoryRazorpayReconstructionStore
    )
    if config.database_url:
        engine = create_async_engine(config.database_url, pool_pre_ping=True)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        repository = SqlHoldRepository(sessions)
        razorpay_event_store = SqlRazorpayEventStore(sessions)
        razorpay_reconstruction_store = SqlRazorpayReconstructionStore(sessions)
        razorpay_timeline_store = SqlPointInTimeTimelineStore(sessions)
        live_feature_store = SqlLiveFeatureStore(sessions)
        checkout_store = SqlCheckoutBridgeStore(sessions)
        shadow_score_store = SqlShadowScoreStore(sessions)
        ground_truth_store = SqlProviderGroundTruthStore(sessions)
        payment_case_reader = SqlRazorpayPaymentCaseReader(sessions)
    else:
        repository = InMemoryHoldRepository()
        razorpay_event_store = InMemoryRazorpayEventStore()
        razorpay_reconstruction_store = InMemoryRazorpayReconstructionStore()
        razorpay_timeline_store = None
        live_feature_store = None
        checkout_store = None
        shadow_score_store = None
        ground_truth_store = None
        payment_case_reader = None
    ledger = AuditLedger(config.ledger_path)
    hold_policy = (
        LocalHoldPolicy() if config.policy_mode == "local" else OpaHoldPolicy(config.opa_url)
    )
    authorizer = (
        LocalAuthorizer() if config.authorization_mode == "local" else OpaAuthorizer(config.opa_url)
    )
    if config.decision_mode == "human_only":
        if config.model_release_path:
            raise ValueError(
                "RAZORTRUST_MODEL_RELEASE_PATH cannot be configured in human_only decision mode"
            )
        risk_model: RiskModel = HumanOnlyRiskModel()
    else:
        risk_model = (
            DevelopmentRiskModel()
            if config.environment == "development"
            else UnavailableRiskModel("no verified production model release is configured")
        )
    evidence_verifier = (
        SignedAttestationEvidenceVerifier(config.evidence_attestation_keys)
        if config.evidence_attestation_keys
        else (
            LegacyEvidenceVerifier()
            if config.environment == "development"
            else UnavailableEvidenceVerifier()
        )
    )
    if config.model_release_path:
        try:
            from .ml.runtime import TrainedRiskModel

            if not config.model_public_key:
                raise ValueError("RAZORTRUST_MODEL_PUBLIC_KEY is required")
            risk_model = TrainedRiskModel.load_release(
                config.model_release_path, public_key_b64=config.model_public_key
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            failure_monitor.record(
                "MODEL_RELEASE_VERIFICATION_FAILED", error_type=type(exc).__name__
            )
            risk_model = UnavailableRiskModel(type(exc).__name__)
    cost_policy = CostPolicy()
    shadow_analysis_runtime = load_shadow_analysis_runtime()
    service = HoldService(
        repository, risk_model, cost_policy, hold_policy, ledger, evidence_verifier
    )
    payment_case_bridge = (
        RazorpayPaymentCaseBridge(payment_case_reader, service)
        if payment_case_reader is not None
        else None
    )
    operator_history = OperatorHistoryStore()
    public_benchmark = PublicBenchmarkService()
    automation_orchestrator = HoldAutomationOrchestrator(repository, service)
    shadow_scoring = (
        ShadowScoringService(shadow_score_store, risk_model, cost_policy)
        if shadow_score_store is not None and live_feature_store is not None
        else None
    )
    layer_execution = (
        LayerExecutionService(
            analysis_runtime=shadow_analysis_runtime,
            cost_policy=cost_policy,
            hold_policy=hold_policy,
            enforcement_mode=config.decision_mode,
            ledger=ledger,
            ground_truth_store=ground_truth_store,
        )
        if live_feature_store is not None
        else None
    )
    razorpay_client = None
    if config.razorpay_key_id and config.razorpay_key_secret:
        try:
            razorpay_client = RazorpayClient(
                key_id=config.razorpay_key_id,
                key_secret=config.razorpay_key_secret,
                base_url=config.razorpay_base_url,
            )
        except ValueError as exc:
            failure_monitor.record(
                "RAZORPAY_CLIENT_CONFIGURATION_INVALID", error_type=type(exc).__name__
            )
    razorpay_reconstructor = (
        RazorpayPaymentReconstructor(
            razorpay_reconstruction_store,
            razorpay_client,
            checkout_lifecycle=checkout_store,
        )
        if razorpay_client is not None and config.database_url
        else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        recovery_task: asyncio.Task[int] | None = None
        checkout_reconcile_task: asyncio.Task[object] | None = None
        automation_task: asyncio.Task[None] | None = None
        automation_stop = asyncio.Event()
        if config.razorpay_enabled and razorpay_reconstructor is not None:
            recovery_task = asyncio.create_task(razorpay_reconstructor.process_pending(limit=100))
        if config.razorpay_enabled and checkout_store is not None and razorpay_client is not None:
            checkout_reconcile_task = asyncio.create_task(
                checkout_store.reconcile_open_orders(razorpay_client, limit=100)
            )
        if config.automation_enabled:
            automation_task = asyncio.create_task(
                automation_orchestrator.run_forever(
                    interval_seconds=config.automation_interval_seconds,
                    stop_event=automation_stop,
                )
            )
        yield
        automation_stop.set()
        for task in (recovery_task, checkout_reconcile_task, automation_task):
            if task is not None and not task.done():
                task.cancel()
        if engine is not None:
            await engine.dispose()

    app = FastAPI(
        title="RazorTrust Settlement Hold API",
        version="0.5.0",
        description=(
            "Point-in-time settlement hold decisions with one evidence round "
            "and fail-safe guardrails."
        ),
        lifespan=lifespan,
    )
    if config.security_headers_enabled:
        app.add_middleware(
            ProductionSecurityMiddleware,
            trusted_hosts=config.trusted_hosts,
            requests_per_minute=config.rate_limit_per_minute,
            hsts=config.environment != "development",
        )
    web_directory = Path(__file__).with_name("web")
    app.mount("/assets", StaticFiles(directory=web_directory), name="assets")
    configure_observability(app, config.environment)

    @app.middleware("http")
    async def trace_context(request: Request, call_next):
        trace_id = uuid4().hex
        token = bind_request_trace_id(trace_id)
        try:
            response = await call_next(request)
            response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            reset_request_trace_id(token)

    async def require_authorization(
        request: Request,
        *,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        merchant_id: str | None = None,
    ) -> Principal:
        principal = _authenticate(request, config)
        try:
            decision = await authorizer.authorize(
                AuthorizationRequest(
                    principal=principal,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    merchant_id=merchant_id,
                )
            )
        except AuthorizationUnavailable as exc:
            failure_monitor.record("AUTHORIZATION_SERVICE_UNAVAILABLE")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="authorization service is unavailable",
            ) from exc
        if not decision.allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="access denied")
        return principal

    @app.exception_handler(HoldNotFound)
    async def hold_not_found(_request, exc: HoldNotFound):
        return _problem(status.HTTP_404_NOT_FOUND, "hold_not_found", str(exc))

    @app.exception_handler(InvalidTransition)
    async def invalid_transition(_request, exc: InvalidTransition):
        return _problem(status.HTTP_409_CONFLICT, "invalid_hold_transition", str(exc))

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(web_directory / "index.html")

    @app.get("/checkout/r4c-test", include_in_schema=False)
    async def checkout_r4c_test() -> FileResponse:
        if not config.razorpay_enabled or config.razorpay_mode != "test":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")
        return FileResponse(web_directory / "checkout-r4c.html")

    @app.get("/health/ready", tags=["health"])
    async def readiness() -> dict[str, str | int]:
        if not evidence_verifier.ready:
            failure_monitor.record("EVIDENCE_VERIFIER_UNAVAILABLE")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="signed evidence verifier is not configured",
            )
        try:
            await repository.healthcheck()
        except SQLAlchemyError as exc:
            failure_monitor.record("DATABASE_HEALTHCHECK_FAILED")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database is unavailable",
            ) from exc
        try:
            valid, records = ledger.verify()
        except AuditFormatError as exc:
            failure_monitor.record("AUDIT_FORMAT_INVALID")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="audit log format is incompatible",
            ) from exc
        if not valid:
            failure_monitor.record("AUDIT_CHAIN_INVALID", first_bad_record=records)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="audit chain is invalid"
            )
        if isinstance(risk_model, UnavailableRiskModel):
            failure_monitor.record("MODEL_RELEASE_UNAVAILABLE")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="configured model release is unavailable or unverified",
            )
        try:
            await hold_policy.healthcheck()
        except PolicyUnavailable as exc:
            failure_monitor.record("POLICY_ENGINE_HEALTHCHECK_FAILED")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="policy service is unavailable",
            ) from exc
        try:
            await authorizer.healthcheck()
        except AuthorizationUnavailable as exc:
            failure_monitor.record("AUTHORIZATION_SERVICE_HEALTHCHECK_FAILED")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="authorization service is unavailable",
            ) from exc
        if config.razorpay_enabled:
            if not config.razorpay_webhook_secret:
                failure_monitor.record("RAZORPAY_WEBHOOK_SECRET_MISSING")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Razorpay webhook secret is not configured",
                )
            if razorpay_client is None:
                failure_monitor.record("RAZORPAY_API_CREDENTIALS_MISSING")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Razorpay API credentials are not configured",
                )
            try:
                await razorpay_event_store.healthcheck()
                await razorpay_reconstruction_store.healthcheck()
                if checkout_store is not None:
                    await checkout_store.healthcheck()
                if shadow_score_store is not None:
                    await shadow_score_store.healthcheck()
            except SQLAlchemyError as exc:
                failure_monitor.record("RAZORPAY_EVENT_STORE_UNAVAILABLE")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Razorpay event store is unavailable",
                ) from exc
        return {
            "status": "ready",
            "decision_mode": config.decision_mode,
            "risk_runtime": risk_model.version,
            "audit_records": records,
            "policy_mode": config.policy_mode,
            "model_version": risk_model.version,
        }

    @app.get("/v1/integrations/razorpay/status", tags=["integrations"])
    async def razorpay_status(request: Request) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="razorpay",
        )
        stats = await razorpay_event_store.stats()
        processing = await razorpay_reconstruction_store.processing_stats()
        checkout_stats = (
            await checkout_store.stats()
            if checkout_store is not None
            else {
                "checkout_sessions": 0,
                "verified_checkout_sessions": 0,
                "checkout_payment_attempts": 0,
                "checkout_open_sessions": 0,
                "checkout_authorized_sessions": 0,
                "checkout_attempted_sessions": 0,
                "checkout_paid_sessions": 0,
                "checkout_abandoned_sessions": 0,
            }
        )
        shadow_stats = await shadow_score_store.stats() if shadow_score_store is not None else None
        return {
            "provider": "RAZORPAY",
            "enabled": config.razorpay_enabled,
            "mode": config.razorpay_mode,
            "webhook_secret_configured": bool(config.razorpay_webhook_secret),
            "api_credentials_configured": razorpay_client is not None,
            "total_events": stats.total_events,
            "last_event_at": (
                stats.last_event_at.isoformat() if stats.last_event_at is not None else None
            ),
            "last_event_type": stats.last_event_type,
            "pending_events": processing.pending_events,
            "processing_events": processing.processing_events,
            "retry_events": processing.retry_events,
            "processed_events": processing.processed_events,
            "failed_events": processing.failed_events,
            "skipped_events": processing.skipped_events,
            "normalized_payments": processing.normalized_payments,
            "normalized_refunds": processing.normalized_refunds,
            "normalized_settlements": processing.normalized_settlements,
            "normalized_disputes": processing.normalized_disputes,
            "reconciliation_items": processing.reconciliation_items,
            "last_processed_at": (
                processing.last_processed_at.isoformat()
                if processing.last_processed_at is not None
                else None
            ),
            "checkout_sessions": checkout_stats["checkout_sessions"],
            "verified_checkout_sessions": checkout_stats["verified_checkout_sessions"],
            "checkout_payment_attempts": checkout_stats["checkout_payment_attempts"],
            "checkout_open_sessions": checkout_stats["checkout_open_sessions"],
            "checkout_authorized_sessions": checkout_stats["checkout_authorized_sessions"],
            "checkout_attempted_sessions": checkout_stats["checkout_attempted_sessions"],
            "checkout_paid_sessions": checkout_stats["checkout_paid_sessions"],
            "checkout_abandoned_sessions": checkout_stats["checkout_abandoned_sessions"],
            "shadow_score_runs": shadow_stats.total_runs if shadow_stats is not None else 0,
            "shadow_scored_runs": shadow_stats.scored_runs if shadow_stats is not None else 0,
            "shadow_blocked_runs": shadow_stats.blocked_runs if shadow_stats is not None else 0,
            "shadow_live_runs": shadow_stats.live_runs if shadow_stats is not None else 0,
            "shadow_fixture_runs": shadow_stats.fixture_runs if shadow_stats is not None else 0,
            "last_shadow_score_at": (
                shadow_stats.last_run_at.isoformat()
                if shadow_stats is not None and shadow_stats.last_run_at is not None
                else None
            ),
        }

    @app.get("/v1/integrations/razorpay/accounts", tags=["integrations"])
    async def razorpay_accounts(
        request: Request, limit: int = Query(default=100, ge=1, le=500)
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="razorpay",
        )
        if razorpay_timeline_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="point-in-time processor timeline requires PostgreSQL",
            )
        accounts = await razorpay_timeline_store.list_accounts(limit=limit)
        return {
            "provider": "RAZORPAY",
            "count": len(accounts),
            "accounts": [item.model_dump(mode="json") for item in accounts],
        }

    @app.get(
        "/v1/integrations/razorpay/accounts/{account_id}/point-in-time",
        tags=["integrations"],
    )
    async def razorpay_point_in_time(
        account_id: str,
        request: Request,
        as_of: datetime | None = Query(default=None),
        lookback_hours: int = Query(default=168, ge=1, le=720),
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="razorpay",
        )
        if razorpay_timeline_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="point-in-time processor timeline requires PostgreSQL",
            )
        cutoff = as_of or datetime.now(UTC)
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="as_of must include a timezone",
            )
        snapshot = await razorpay_timeline_store.snapshot(
            account_id=account_id,
            as_of=cutoff.astimezone(UTC),
            lookback_hours=lookback_hours,
        )
        return snapshot.model_dump(mode="json")

    @app.post(
        "/v1/integrations/razorpay/checkout/orders",
        response_model=CheckoutOrderResponse,
        tags=["integrations"],
    )
    async def create_razorpay_checkout_order(
        payload: CheckoutOrderCreate, request: Request
    ) -> CheckoutOrderResponse:
        principal_hint = _authenticate(request, config)
        merchant_hint = (
            principal_hint.merchant_id if principal_hint.role == ActorRole.MERCHANT else None
        )
        await require_authorization(
            request,
            action="CREATE_CHECKOUT",
            resource_type="INTEGRATION",
            resource_id="razorpay-checkout",
            merchant_id=merchant_hint,
        )
        if not config.razorpay_enabled or config.razorpay_mode != "test":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="R4C checkout bridge is restricted to Razorpay Test Mode",
            )
        if razorpay_client is None or checkout_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Razorpay checkout bridge is unavailable",
            )
        try:
            account_id = await checkout_store.resolve_account_id(preferred_account_id=merchant_hint)
            pending = await checkout_store.create_pending(
                account_id=account_id,
                amount=payload.amount,
                currency=payload.currency,
                device_pseudonym=payload.device_pseudonym,
                customer_geo=payload.customer_geo,
                client_event_at=payload.client_event_at,
            )
            provider_order = await razorpay_client.create_order(
                amount=pending.amount,
                currency=pending.currency,
                receipt=pending.receipt,
                notes={
                    "purpose": "razortrust-r4c-shadow",
                    "checkout_session": str(pending.checkout_session_id),
                },
            )
            finalized = await checkout_store.finalize_provider_order(
                checkout_session_id=pending.checkout_session_id,
                provider_order=provider_order,
            )
        except LookupError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except RazorpayApiError as exc:
            if "pending" in locals():
                await checkout_store.mark_provider_failed(pending.checkout_session_id)
            failure_monitor.record(
                "RAZORPAY_CHECKOUT_ORDER_API_FAILED",
                status_code=exc.status_code or 0,
            )
            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                    if exc.retryable
                    else status.HTTP_502_BAD_GATEWAY
                ),
                detail="Razorpay order creation failed",
            ) from exc
        except ValueError as exc:
            failure_monitor.record("RAZORPAY_CHECKOUT_ORDER_CONTRACT_FAILED")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        if finalized.razorpay_order_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Razorpay order was not persisted",
            )
        return CheckoutOrderResponse(
            checkout_session_id=finalized.checkout_session_id,
            razorpay_order_id=finalized.razorpay_order_id,
            amount=finalized.amount,
            currency=finalized.currency,
            key_id=config.razorpay_key_id or "",
            provider_status=finalized.provider_status,
        )

    @app.post(
        "/v1/integrations/razorpay/checkout/verify",
        response_model=CheckoutVerificationResponse,
        tags=["integrations"],
    )
    async def verify_razorpay_checkout(
        payload: CheckoutVerificationSubmission, request: Request
    ) -> CheckoutVerificationResponse:
        if not config.razorpay_enabled or config.razorpay_mode != "test":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="R4C checkout bridge is restricted to Razorpay Test Mode",
            )
        if razorpay_client is None or checkout_store is None or not config.razorpay_key_secret:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Razorpay checkout verification is unavailable",
            )
        try:
            checkout = await checkout_store.get_session(payload.checkout_session_id)
        except LookupError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        await require_authorization(
            request,
            action="VERIFY_CHECKOUT",
            resource_type="INTEGRATION",
            resource_id=str(payload.checkout_session_id),
            merchant_id=checkout.account_id,
        )
        if checkout.razorpay_order_id is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="checkout provider order is not ready",
            )
        if payload.razorpay_order_id != checkout.razorpay_order_id:
            failure_monitor.record("RAZORPAY_CHECKOUT_ORDER_ID_MISMATCH")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="checkout verification failed",
            )
        if not verify_checkout_signature(
            order_id=checkout.razorpay_order_id,
            payment_id=payload.razorpay_payment_id,
            signature=payload.razorpay_signature,
            secret=config.razorpay_key_secret,
        ):
            failure_monitor.record("RAZORPAY_CHECKOUT_SIGNATURE_INVALID")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="checkout verification failed",
            )
        try:
            payment = await razorpay_client.fetch_payment(payload.razorpay_payment_id)
            authoritative_status = validate_authoritative_payment(
                stored_order_id=checkout.razorpay_order_id,
                stored_amount=checkout.amount,
                stored_currency=checkout.currency,
                payment_id=payload.razorpay_payment_id,
                payment=payment,
            )
            await checkout_store.bind_verified_payment(
                checkout_session_id=checkout.checkout_session_id,
                payment_id=payload.razorpay_payment_id,
                authoritative_status=authoritative_status,
            )
        except RazorpayApiError as exc:
            failure_monitor.record(
                "RAZORPAY_CHECKOUT_PAYMENT_FETCH_FAILED",
                status_code=exc.status_code or 0,
            )
            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                    if exc.retryable
                    else status.HTTP_502_BAD_GATEWAY
                ),
                detail="authoritative payment verification failed",
            ) from exc
        except ValueError as exc:
            failure_monitor.record("RAZORPAY_CHECKOUT_PAYMENT_CONTRACT_FAILED")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="authoritative payment verification failed",
            ) from exc
        captured = authoritative_status == "captured" and payment.get("captured") is True
        return CheckoutVerificationResponse(
            status="verified",
            checkout_session_id=checkout.checkout_session_id,
            razorpay_order_id=checkout.razorpay_order_id,
            razorpay_payment_id=payload.razorpay_payment_id,
            signature_verified=True,
            authoritative_payment_status=authoritative_status,
            telemetry_bound=True,
            fulfillment_eligible=captured,
        )

    @app.post("/v1/integrations/razorpay/checkout/reconcile", tags=["integrations"])
    async def reconcile_razorpay_checkout_lifecycle(
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
        abandon_after_hours: int = Query(default=24, ge=1, le=720),
    ) -> dict[str, int | str]:
        await require_authorization(
            request,
            action="MANAGE_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="razorpay-checkout",
        )
        if checkout_store is None or razorpay_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Razorpay checkout lifecycle reconciliation is unavailable",
            )
        result = await checkout_store.reconcile_open_orders(
            razorpay_client,
            limit=limit,
            abandon_after_hours=abandon_after_hours,
        )
        return {
            "status": "ok",
            "inspected": result.inspected,
            "updated": result.updated,
            "paid": result.paid,
            "attempted": result.attempted,
            "open": result.open,
            "abandoned": result.abandoned,
            "errors": result.errors,
        }

    @app.post("/v1/integrations/razorpay/telemetry", tags=["integrations"])
    async def razorpay_first_party_telemetry(
        submission: FirstPartyTelemetrySubmission,
        request: Request,
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="SUBMIT_TELEMETRY",
            resource_type="INTEGRATION",
            resource_id="razorpay",
            merchant_id=submission.account_id,
        )
        if live_feature_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="first-party telemetry requires PostgreSQL",
            )
        try:
            receipt = await live_feature_store.submit_telemetry(submission)
        except LookupError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return receipt.model_dump(mode="json")

    @app.get(
        "/v1/integrations/razorpay/accounts/{account_id}/feature-contract",
        tags=["integrations"],
    )
    async def razorpay_feature_contract(
        account_id: str,
        request: Request,
        as_of: datetime | None = Query(default=None),
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_FEATURE_CONTRACT",
            resource_type="INTEGRATION",
            resource_id="razorpay",
        )
        if live_feature_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="live feature contract requires PostgreSQL",
            )
        cutoff = as_of or datetime.now(UTC)
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="as_of must include a timezone",
            )
        preview = await live_feature_store.feature_contract_preview(
            account_id=account_id,
            as_of=cutoff.astimezone(UTC),
        )
        return preview.model_dump(mode="json")

    @app.get(
        "/v1/integrations/razorpay/accounts/{account_id}/shadow-score",
        tags=["integrations"],
    )
    async def razorpay_shadow_score(
        account_id: str,
        request: Request,
        as_of: datetime | None = Query(default=None),
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_FEATURE_CONTRACT",
            resource_type="INTEGRATION",
            resource_id="razorpay",
        )
        if live_feature_store is None or shadow_scoring is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="shadow scoring requires PostgreSQL and a configured risk runtime",
            )
        cutoff = as_of or datetime.now(UTC)
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="as_of must include a timezone",
            )
        preview = await live_feature_store.feature_contract_preview(
            account_id=account_id,
            as_of=cutoff.astimezone(UTC),
        )
        result = await shadow_scoring.score_live_preview(preview)
        return result.model_dump(mode="json")

    @app.get("/v1/public-benchmark/ulb/status", tags=["public-benchmark"])
    async def public_benchmark_status(request: Request) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="public-benchmark-ulb",
        )
        return public_benchmark.status().model_dump(mode="json")

    @app.post("/v1/public-benchmark/ulb/prepare", tags=["public-benchmark"])
    async def prepare_public_benchmark(
        request: Request,
        force: bool = Query(default=False),
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="public-benchmark-ulb",
        )
        try:
            result = public_benchmark.prepare(force=force)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"public benchmark preparation failed: {exc}",
            ) from exc
        return result.model_dump(mode="json")

    @app.get("/v1/public-benchmark/ulb/cases", tags=["public-benchmark"])
    async def public_benchmark_cases(
        request: Request,
        label: str = Query(default="fraud"),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="public-benchmark-ulb",
        )
        try:
            items = public_benchmark.list_cases(label=label, limit=limit)
        except (LookupError, ValueError) as exc:
            code = (
                status.HTTP_409_CONFLICT
                if isinstance(exc, LookupError)
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(status_code=code, detail=str(exc)) from exc
        return {
            "count": len(items),
            "items": [item.model_dump(mode="json") for item in items],
            "research_only": True,
            "production_action_eligible": False,
        }

    @app.post(
        "/v1/public-benchmark/ulb/cases/{row_id}/execute",
        tags=["public-benchmark"],
    )
    async def execute_public_benchmark_case(
        row_id: int,
        request: Request,
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id=f"public-benchmark-ulb:{row_id}",
        )
        try:
            result = public_benchmark.execute(row_id)
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        return result.model_dump(mode="json")

    @app.post("/v1/operator-history/audit-csv", tags=["operator-history"])
    async def audit_operator_history_csv(request: Request) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="operator-history",
        )
        try:
            report = audit_csv_bytes(await request.body())
        except HistoryImportValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return report.model_dump(mode="json")

    @app.post("/v1/operator-history/import", tags=["operator-history"])
    async def import_operator_history_csv(
        request: Request,
        dataset_name: str = Query(min_length=1, max_length=160),
        account_id: str = Query(min_length=1, max_length=128),
        source_kind: str = Query(default="OPERATOR_SUPPLIED_REAL_HISTORY"),
        source_description: str = Query(min_length=1, max_length=500),
        source_url: str | None = Query(default=None, max_length=1000),
        user_attested_real_data: bool = Query(default=False),
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="CREATE_HOLD",
            resource_type="HOLD",
            resource_id="operator-history-import",
        )
        try:
            manifest = operator_history.import_csv_bytes(
                await request.body(),
                dataset_name=dataset_name,
                account_id=account_id,
                source_kind=source_kind,
                source_description=source_description,
                source_url=source_url,
                user_attested_real_data=user_attested_real_data,
            )
        except HistoryImportValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return manifest.model_dump(mode="json")

    @app.get("/v1/operator-history/datasets", tags=["operator-history"])
    async def list_operator_history_datasets(request: Request) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="operator-history",
        )
        items = operator_history.list_manifests()
        return {"count": len(items), "items": [item.model_dump(mode="json") for item in items]}

    @app.get("/v1/operator-history/candidates", tags=["operator-history"])
    async def operator_history_candidates(
        request: Request,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="operator-history",
        )
        items = operator_history.candidates(limit=limit)
        return {
            "count": len(items),
            "items": [item.model_dump(mode="json") for item in items],
            "provider_verified": False,
            "synthetic_fixture": False,
        }

    @app.post(
        "/v1/operator-history/{dataset_id}/transactions/{transaction_id}/layer-execution",
        tags=["operator-history", "holds"],
    )
    async def execute_operator_history_case(
        dataset_id: str,
        transaction_id: str,
        request: Request,
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="CREATE_HOLD",
            resource_type="HOLD",
            resource_id=f"{dataset_id}:{transaction_id}",
        )
        if layer_execution is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="layer execution is unavailable",
            )
        try:
            prepared = await operator_history.prepare_case(
                dataset_id,
                transaction_id,
                service,
            )
            result = await layer_execution.execute(
                hold=prepared.hold,
                preview=prepared.preview,
                context=prepared.context,
            )
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except HistoryImportValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return result.model_dump(mode="json")

    @app.get(
        "/v1/integrations/razorpay/payment-candidates",
        tags=["integrations"],
    )
    async def razorpay_payment_candidates(
        request: Request,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="razorpay",
        )
        if payment_case_reader is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="authoritative Razorpay payment store is unavailable",
            )
        try:
            candidates = await payment_case_reader.list_candidates(limit=limit)
        except PaymentCaseBridgeUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="authoritative Razorpay payment store is unavailable",
            ) from exc
        return {
            "provider": "RAZORPAY",
            "source": "CANONICAL_PROCESSOR_STORE",
            "count": len(candidates),
            "items": [item.model_dump(mode="json") for item in candidates],
            "synthetic_fixture": False,
        }

    @app.post(
        "/v1/integrations/razorpay/payments/{payment_id}/case",
        tags=["integrations", "holds"],
    )
    async def create_payment_linked_case(
        payment_id: str,
        request: Request,
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="CREATE_HOLD",
            resource_type="HOLD",
            resource_id=payment_id,
        )
        if payment_case_bridge is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="real payment case bridge requires PostgreSQL",
            )
        try:
            receipt = await payment_case_bridge.create_or_get(payment_id)
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except (PaymentCaseBridgeUnavailable, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="real payment case bridge is unavailable",
            ) from exc
        return receipt.model_dump(mode="json")

    @app.post(
        "/v1/holds/{hold_id}/layer-execution",
        tags=["holds"],
    )
    async def execute_hold_layers(
        hold_id: UUID,
        request: Request,
        as_of: datetime | None = Query(default=None),
    ) -> dict[str, object]:
        hold = await repository.get(hold_id)
        await require_authorization(
            request,
            action="EVALUATE_HOLD",
            resource_type="HOLD",
            resource_id=str(hold_id),
            merchant_id=hold.merchant_id,
        )
        if live_feature_store is None or layer_execution is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="layer execution requires PostgreSQL live history",
            )
        cutoff = as_of or hold.triggered_at
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="as_of must include a timezone",
            )
        preview = await live_feature_store.feature_contract_preview(
            account_id=hold.merchant_id,
            as_of=cutoff.astimezone(UTC),
        )
        result = await layer_execution.execute(hold=hold, preview=preview)
        return result.model_dump(mode="json")

    @app.post(
        "/v1/integrations/razorpay/shadow-score/fixture",
        tags=["integrations"],
    )
    async def razorpay_shadow_score_fixture(
        request: Request,
        fixture_name: str = Query(default="normal_baseline"),
    ) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_FEATURE_CONTRACT",
            resource_type="INTEGRATION",
            resource_id="razorpay",
        )
        if config.razorpay_mode != "test":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="synthetic fixture scoring is available only in Razorpay Test Mode",
            )
        if shadow_scoring is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="shadow scoring requires PostgreSQL and a configured risk runtime",
            )
        if fixture_name not in {"normal_baseline", "novel_risk_burst"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="fixture_name must be normal_baseline or novel_risk_burst",
            )
        result = await shadow_scoring.score_fixture(
            fixture_name=fixture_name,
            as_of=datetime.now(UTC),
        )
        return result.model_dump(mode="json")

    @app.post("/v1/integrations/razorpay/webhook", tags=["integrations"])
    async def razorpay_webhook(
        request: Request, background_tasks: BackgroundTasks
    ) -> dict[str, object]:
        if not config.razorpay_enabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Razorpay integration is disabled",
            )
        if not config.razorpay_webhook_secret:
            failure_monitor.record("RAZORPAY_WEBHOOK_SECRET_MISSING")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Razorpay webhook secret is not configured",
            )

        raw_body = await request.body()
        signature = request.headers.get("X-Razorpay-Signature", "")
        provider_event_id = request.headers.get("x-razorpay-event-id", "").strip()
        if not provider_event_id or len(provider_event_id) > 128:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="x-razorpay-event-id is required and must be <= 128 characters",
            )
        if not verify_webhook_signature(raw_body, signature, config.razorpay_webhook_secret):
            failure_monitor.record(
                "RAZORPAY_WEBHOOK_SIGNATURE_INVALID", provider_event_id=provider_event_id
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid Razorpay webhook signature",
            )

        try:
            decoded = json.loads(raw_body)
            envelope = RazorpayWebhookEnvelope.model_validate(decoded)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid Razorpay webhook payload",
            ) from exc

        received_at = datetime.now(UTC)
        stored, created = await razorpay_event_store.store(
            provider_event_id=provider_event_id,
            envelope=envelope,
            payload_sha256=payload_sha256(raw_body),
            summary=build_event_summary(envelope),
            received_at=received_at,
        )
        if (created or stored.processing_status in {"RECEIVED", "RETRY"}) and (
            razorpay_reconstructor is not None
        ):
            background_tasks.add_task(razorpay_reconstructor.process_event, stored.event_id)
        return {
            "status": "accepted",
            "duplicate": not created,
            "provider_event_id": stored.provider_event_id,
            "event_type": stored.event_type,
            "processing_status": stored.processing_status,
        }

    @app.post("/v1/integrations/razorpay/process-pending", tags=["integrations"])
    async def process_pending_razorpay_events(
        request: Request, limit: int = Query(default=100, ge=1, le=1000)
    ) -> dict[str, int | str]:
        await require_authorization(
            request,
            action="MANAGE_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="razorpay",
        )
        if razorpay_reconstructor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Razorpay payment reconstruction is unavailable",
            )
        attempted = await razorpay_reconstructor.process_pending(limit=limit)
        return {"status": "ok", "attempted_events": attempted}

    @app.post("/v1/integrations/razorpay/reconciliation/sync", tags=["integrations"])
    async def sync_razorpay_reconciliation(
        request: Request,
        year: int = Query(ge=2000, le=2200),
        month: int = Query(ge=1, le=12),
        day: int | None = Query(default=None, ge=1, le=31),
        page_size: int = Query(default=200, ge=1, le=1000),
        max_pages: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, int | str]:
        await require_authorization(
            request,
            action="MANAGE_INTEGRATION",
            resource_type="INTEGRATION",
            resource_id="razorpay",
        )
        if razorpay_reconstructor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Razorpay reconstruction is unavailable",
            )
        try:
            result = await razorpay_reconstructor.sync_reconciliation(
                year=year,
                month=month,
                day=day,
                page_size=page_size,
                max_pages=max_pages,
            )
        except RazorpayApiError as exc:
            failure_monitor.record(
                "RAZORPAY_RECONCILIATION_API_FAILED",
                status_code=exc.status_code or 0,
            )
            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                    if exc.retryable
                    else status.HTTP_502_BAD_GATEWAY
                ),
                detail="Razorpay reconciliation API is unavailable",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Razorpay reconciliation response is invalid",
            ) from exc
        return {
            "status": "ok",
            "fetched_items": result.fetched_items,
            "stored_items": result.stored_items,
            "pages": result.pages,
        }

    @app.post("/v1/holds", response_model=HoldCase, tags=["holds"])
    async def create_hold(payload: HoldCreate, request: Request, response: Response) -> HoldCase:
        await require_authorization(
            request,
            action="CREATE_HOLD",
            resource_type="HOLD",
            merchant_id=payload.merchant_id,
        )
        hold, created = await service.create_hold(payload)
        response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        return hold

    @app.get("/v1/holds", response_model=list[HoldCase], tags=["holds"])
    async def list_holds(
        request: Request,
        merchant_id: str | None = Query(default=None, min_length=1, max_length=128),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> list[HoldCase]:
        await require_authorization(
            request,
            action="LIST_HOLDS",
            resource_type="HOLD_COLLECTION",
            merchant_id=merchant_id,
        )
        return await repository.list_holds(merchant_id=merchant_id, limit=limit)

    @app.get("/v1/operator/summary", tags=["operator"])
    async def operator_summary(request: Request) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_OPERATOR_DASHBOARD",
            resource_type="OPERATOR_DASHBOARD",
        )
        holds = await repository.list_holds(merchant_id=None, limit=200)
        decisions: list[DecisionResponse] = []
        for hold in holds:
            try:
                decisions.append(await repository.get_decision(hold.hold_id))
            except InvalidTransition:
                continue
        decision_counts = {
            decision.value: sum(item.decision == decision for item in decisions)
            for decision in HoldDecision
        }
        risk_family_counts: dict[str, int] = {}
        for hold in holds:
            family = hold.reason_code or "UNSPECIFIED"
            risk_family_counts[family] = risk_family_counts.get(family, 0) + 1
        evidence_complete = sum(hold.evidence_round == 1 for hold in holds)
        pending = sum(hold.state == "HUMAN_REVIEW" for hold in holds)
        return {
            "case_count": len(holds),
            "pending_human_reviews": pending,
            "decision_counts": decision_counts,
            "risk_family_counts": dict(
                sorted(risk_family_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "evidence_completeness": (round(evidence_complete / len(holds), 4) if holds else 0.0),
            "recent_decisions": [
                {
                    "hold_id": str(item.hold_id),
                    "decision": item.decision,
                    "created_at": item.created_at,
                }
                for item in sorted(decisions, key=lambda value: value.created_at, reverse=True)[:10]
            ],
            "research": FINAL_V3_RESEARCH_STATUS,
            "phases": PHASE_IMPLEMENTATION_STATUS,
            "model_governance": model_governance_status(decision_mode=config.decision_mode),
            "automation": {
                "webhook_ingestion": bool(config.razorpay_enabled),
                "payment_reconstruction": razorpay_reconstructor is not None,
                "recovery_on_startup": bool(
                    config.razorpay_enabled and razorpay_reconstructor is not None
                ),
                "checkout_reconciliation_on_startup": bool(
                    config.razorpay_enabled
                    and checkout_store is not None
                    and razorpay_client is not None
                ),
                "risk_decision_mode": config.decision_mode,
                "automatic_hold_evaluation": config.automation_enabled,
                "automation_interval_seconds": config.automation_interval_seconds,
                "automatic_release": False if config.decision_mode == "human_only" else None,
                "human_authorization_required": True,
            },
            "system_health": "READY",
        }

    @app.get("/v1/operator/model-governance", tags=["operator"])
    async def operator_model_governance(request: Request) -> dict[str, object]:
        await require_authorization(
            request,
            action="VIEW_OPERATOR_DASHBOARD",
            resource_type="OPERATOR_DASHBOARD",
        )
        return model_governance_status(decision_mode=config.decision_mode)

    @app.post("/v1/holds/{hold_id}/evaluate", response_model=DecisionResponse, tags=["holds"])
    async def evaluate_hold(
        hold_id: UUID, payload: HoldEvaluationInput, request: Request
    ) -> DecisionResponse:
        hold = await repository.get(hold_id)
        await require_authorization(
            request,
            action="EVALUATE_HOLD",
            resource_type="HOLD",
            resource_id=str(hold_id),
            merchant_id=hold.merchant_id,
        )
        return await service.evaluate(hold_id, payload)

    @app.get("/v1/holds/{hold_id}", response_model=HoldCase, tags=["holds"])
    async def get_hold(hold_id: UUID, request: Request) -> HoldCase:
        hold = await repository.get(hold_id)
        await require_authorization(
            request,
            action="VIEW_HOLD",
            resource_type="HOLD",
            resource_id=str(hold_id),
            merchant_id=hold.merchant_id,
        )
        return hold

    @app.get("/v1/holds/{hold_id}/decision", response_model=DecisionResponse, tags=["holds"])
    async def get_decision(hold_id: UUID, request: Request) -> DecisionResponse:
        hold = await repository.get(hold_id)
        await require_authorization(
            request,
            action="VIEW_HOLD",
            resource_type="HOLD",
            resource_id=str(hold_id),
            merchant_id=hold.merchant_id,
        )
        return await repository.get_decision(hold_id)

    @app.post("/v1/holds/{hold_id}/evidence", response_model=EvidenceRecord, tags=["evidence"])
    async def submit_evidence(
        hold_id: UUID, payload: EvidenceSubmission, request: Request
    ) -> EvidenceRecord:
        hold = await repository.get(hold_id)
        await require_authorization(
            request,
            action="SUBMIT_EVIDENCE",
            resource_type="HOLD",
            resource_id=str(hold_id),
            merchant_id=hold.merchant_id,
        )
        return await service.submit_evidence(hold_id, payload)

    @app.post("/v1/holds/{hold_id}/rescore", response_model=DecisionResponse, tags=["evidence"])
    async def rescore_hold(hold_id: UUID, request: Request) -> DecisionResponse:
        hold = await repository.get(hold_id)
        await require_authorization(
            request,
            action="EVALUATE_HOLD",
            resource_type="HOLD",
            resource_id=str(hold_id),
            merchant_id=hold.merchant_id,
        )
        return await service.rescore(hold_id)

    @app.get("/v1/holds/{hold_id}/audit", tags=["audit"])
    async def get_audit(hold_id: UUID, request: Request) -> list[dict]:
        hold = await repository.get(hold_id)
        await require_authorization(
            request,
            action="VIEW_AUDIT",
            resource_type="HOLD",
            resource_id=str(hold_id),
            merchant_id=hold.merchant_id,
        )
        if isinstance(repository, TransactionalAuditRepository):
            return await repository.get_audit_records(hold_id)
        return ledger.records_for_case(hold_id)

    @app.get("/v1/holds/{hold_id}/dossier.pdf", tags=["audit"])
    async def get_evidence_dossier(hold_id: UUID, request: Request) -> FastAPIResponse:
        hold = await repository.get(hold_id)
        await require_authorization(
            request,
            action="VIEW_AUDIT",
            resource_type="HOLD",
            resource_id=str(hold_id),
            merchant_id=hold.merchant_id,
        )
        decision = await repository.get_decision(hold_id)
        records = (
            await repository.get_audit_records(hold_id)
            if isinstance(repository, TransactionalAuditRepository)
            else ledger.records_for_case(hold_id)
        )
        content = build_evidence_dossier(hold, decision, records)
        return FastAPIResponse(
            content=content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="razortrust-{hold_id}.pdf"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/v1/holds/{hold_id}/evidence-case", tags=["audit"])
    async def get_evidence_case(hold_id: UUID, request: Request) -> dict[str, object]:
        hold = await repository.get(hold_id)
        await require_authorization(
            request,
            action="VIEW_AUDIT",
            resource_type="HOLD",
            resource_id=str(hold_id),
            merchant_id=hold.merchant_id,
        )
        evaluation = await repository.get_evaluation_input(hold_id)
        try:
            evidence = await repository.get_evidence(hold_id)
        except InvalidTransition:
            evidence = None
        transactions = sorted(evaluation.transactions, key=lambda item: item.timestamp)
        return {
            "transactions": [
                {
                    "transaction_id": item.transaction_id,
                    "timestamp": item.timestamp,
                    "amount": item.amount,
                    "auth_status": item.auth_status,
                    "device_fingerprint": item.device_fingerprint,
                    "device_status": (
                        "KNOWN"
                        if item.device_fingerprint in evaluation.baseline.known_devices
                        else "NEW"
                    ),
                    "customer_geo": item.customer_geo,
                    "geo_status": (
                        "KNOWN" if item.customer_geo in evaluation.baseline.known_geos else "NEW"
                    ),
                }
                for item in transactions[-25:]
            ],
            "device_summary": {
                "known": sum(
                    item.device_fingerprint in evaluation.baseline.known_devices
                    for item in transactions
                ),
                "new": sum(
                    item.device_fingerprint not in evaluation.baseline.known_devices
                    for item in transactions
                ),
            },
            "geo_summary": {
                "known": sum(
                    item.customer_geo in evaluation.baseline.known_geos for item in transactions
                ),
                "new": sum(
                    item.customer_geo not in evaluation.baseline.known_geos for item in transactions
                ),
            },
            "supporting_document": (
                {
                    "evidence_type": evidence.submission.evidence_type,
                    "content_sha256": evidence.submission.content_sha256,
                    "submitted_at": evidence.submission.submitted_at,
                    "observed_at": evidence.submission.evidence_observed_at,
                    "type_match": evidence.type_match,
                    "recency_hours": evidence.recency_hours,
                }
                if evidence is not None
                else None
            ),
        }

    @app.post(
        "/v1/holds/{hold_id}/analyst-outcome",
        response_model=AnalystReviewRecord,
        tags=["human-review"],
    )
    async def record_analyst_outcome(
        hold_id: UUID, payload: AnalystReviewSubmission, request: Request
    ) -> AnalystReviewRecord:
        hold = await repository.get(hold_id)
        principal = await require_authorization(
            request,
            action="RECORD_ANALYST_OUTCOME",
            resource_type="HOLD",
            resource_id=str(hold_id),
            merchant_id=hold.merchant_id,
        )
        return await service.record_analyst_review(hold_id, payload, analyst_id=principal.id)

    @app.post("/v1/internal/policy/check", response_model=PolicyResult, tags=["internal"])
    async def check_policy(payload: PolicyCheck, request: Request) -> PolicyResult:
        await require_authorization(
            request,
            action="CHECK_POLICY",
            resource_type="POLICY",
            resource_id=payload.policy_version,
        )
        try:
            return await hold_policy.check(payload)
        except PolicyUnavailable:
            failure_monitor.record("POLICY_ENGINE_UNAVAILABLE")
            return PolicyResult(
                allowed_decision=HoldDecision.ESCALATE,
                guardrail_triggered=True,
                reasons=["POLICY_ENGINE_UNAVAILABLE"],
                policy_version="fail-safe@1",
            )

    return app


def _problem(status_code: int, code: str, detail: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"code": code, "detail": detail})


def _authenticate(request: Request, settings: Settings) -> Principal:
    if not settings.authorization_required:
        return Principal(id="development-admin", role=ActorRole.ADMIN)
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required"
        )
    token_sha256 = hashlib.sha256(token.encode()).hexdigest()
    principal_data = settings.api_principals.get(token_sha256)
    if principal_data is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
    try:
        return Principal.model_validate(principal_data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="configured principal is invalid",
        ) from exc


app = create_app()
