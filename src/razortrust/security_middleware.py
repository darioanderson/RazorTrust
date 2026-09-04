from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterable

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class ProductionSecurityMiddleware:
    """Small single-process safety boundary; edge limits remain mandatory at scale."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        trusted_hosts: Iterable[str],
        requests_per_minute: int,
        hsts: bool,
    ) -> None:
        self.app = app
        self.trusted_hosts = {host.lower() for host in trusted_hosts}
        self.limit = requests_per_minute
        self.hsts = hsts
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        host = headers.get(b"host", b"").decode("latin-1").split(":", 1)[0].lower()
        if "*" not in self.trusted_hosts and host not in self.trusted_hosts:
            await self._reject(send, 400, "untrusted_host")
            return
        client = scope.get("client")
        client_id = str(client[0]) if client else "unknown"
        if self.limit > 0 and not self._allow(client_id):
            await self._reject(send, 429, "rate_limit_exceeded", retry_after="60")
            return

        async def secure_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                        (
                            b"content-security-policy",
                            b"default-src 'self'; frame-ancestors 'none'; base-uri 'self'; "
                            b"form-action 'self'",
                        ),
                        (b"cache-control", b"no-store"),
                    ]
                )
                if self.hsts:
                    response_headers.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )
                message["headers"] = response_headers
            await send(message)

        await self.app(scope, receive, secure_send)

    def _allow(self, client_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            bucket = self._requests[client_id]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True

    @staticmethod
    async def _reject(
        send: Send, status: int, code: str, *, retry_after: str | None = None
    ) -> None:
        body = json.dumps({"type": code, "status": status}).encode()
        headers: list[tuple[bytes, bytes]] = [(b"content-type", b"application/problem+json")]
        if retry_after:
            headers.append((b"retry-after", retry_after.encode()))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})


def parse_trusted_hosts(raw: str) -> tuple[str, ...]:
    hosts = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    if not hosts:
        raise ValueError("RAZORTRUST_TRUSTED_HOSTS must contain at least one host")
    return hosts
