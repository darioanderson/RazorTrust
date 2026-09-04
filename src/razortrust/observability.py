from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

_request_trace_id: ContextVar[str | None] = ContextVar("razortrust_trace_id", default=None)


def bind_request_trace_id(trace_id: str) -> Token[str | None]:
    return _request_trace_id.set(trace_id)


def reset_request_trace_id(token: Token[str | None]) -> None:
    _request_trace_id.reset(token)


@contextmanager
def domain_span(name: str, attributes: dict[str, str | int | float | bool]) -> Iterator[None]:
    """Create a safe domain span when OpenTelemetry is installed; otherwise do nothing."""
    try:
        from opentelemetry import trace
    except ImportError:
        yield
        return
    with trace.get_tracer("razortrust.domain").start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield


def current_trace_id(fallback: str) -> str:
    try:
        from opentelemetry import trace
    except ImportError:
        return _request_trace_id.get() or fallback
    trace_id = trace.get_current_span().get_span_context().trace_id
    return f"{trace_id:032x}" if trace_id else (_request_trace_id.get() or fallback)


def configure_observability(app: Any, environment: str) -> None:
    """Enable Sentry and OTLP tracing only when configured and installed."""
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        try:
            import sentry_sdk
        except ImportError:
            warnings.warn(
                "SENTRY_DSN is set but the observability extra is not installed", stacklevel=2
            )
        else:
            sentry_sdk.init(
                dsn=sentry_dsn,
                environment=environment,
                traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
                send_default_pii=False,
            )

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        return
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace import set_tracer_provider
    except ImportError:
        warnings.warn(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but the observability extra is not installed",
            stacklevel=2,
        )
        return

    provider = TracerProvider(resource=Resource.create({"service.name": "razortrust-api"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
