from __future__ import annotations

import logging
from collections import Counter
from typing import Protocol

logger = logging.getLogger(__name__)


class AlertSink(Protocol):
    def critical(self, code: str, context: dict[str, str | int]) -> None: ...


class LoggingAlertSink:
    def critical(self, code: str, context: dict[str, str | int]) -> None:
        logger.critical("razortrust critical failure: %s %s", code, context)


class SentryAlertSink:
    def critical(self, code: str, context: dict[str, str | int]) -> None:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for key, value in context.items():
                scope.set_extra(key, value)
            sentry_sdk.capture_message(f"RazorTrust critical failure: {code}", level="fatal")


class CriticalFailureMonitor:
    def __init__(self, sink: AlertSink, *, repeat_threshold: int = 3) -> None:
        if repeat_threshold < 1:
            raise ValueError("repeat_threshold must be positive")
        self.sink = sink
        self.repeat_threshold = repeat_threshold
        self.counts: Counter[str] = Counter()

    def record(self, code: str, **context: str | int) -> bool:
        self.counts[code] += 1
        count = self.counts[code]
        if count == 1 or count % self.repeat_threshold == 0:
            self.sink.critical(code, {**context, "occurrences": count})
            return True
        return False
