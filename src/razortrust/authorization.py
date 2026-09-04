from __future__ import annotations

from typing import Protocol

import httpx

from .domain import ActorRole, AuthorizationDecision, AuthorizationRequest


class AuthorizationUnavailable(RuntimeError):
    pass


class Authorizer(Protocol):
    async def healthcheck(self) -> None: ...

    async def authorize(self, request: AuthorizationRequest) -> AuthorizationDecision: ...


class LocalAuthorizer:
    version = "api-authz-local@1"
    _roles_by_action = {
        "CREATE_HOLD": {ActorRole.MERCHANT, ActorRole.RISK_SERVICE, ActorRole.ADMIN},
        "EVALUATE_HOLD": {ActorRole.RISK_ANALYST, ActorRole.RISK_SERVICE, ActorRole.ADMIN},
        "VIEW_HOLD": {
            ActorRole.MERCHANT,
            ActorRole.RISK_ANALYST,
            ActorRole.RISK_SERVICE,
            ActorRole.ADMIN,
        },
        "LIST_HOLDS": {
            ActorRole.MERCHANT,
            ActorRole.RISK_ANALYST,
            ActorRole.RISK_SERVICE,
            ActorRole.ADMIN,
        },
        "SUBMIT_EVIDENCE": {
            ActorRole.MERCHANT,
            ActorRole.EVIDENCE_SERVICE,
            ActorRole.ADMIN,
        },
        "VIEW_AUDIT": {ActorRole.RISK_ANALYST, ActorRole.ADMIN},
        "VIEW_OPERATOR_DASHBOARD": {ActorRole.RISK_ANALYST, ActorRole.ADMIN},
        "CHECK_POLICY": {ActorRole.RISK_SERVICE, ActorRole.ADMIN},
        "RECORD_ANALYST_OUTCOME": {ActorRole.RISK_ANALYST, ActorRole.ADMIN},
        "VIEW_INTEGRATION": {ActorRole.RISK_ANALYST, ActorRole.RISK_SERVICE, ActorRole.ADMIN},
        "MANAGE_INTEGRATION": {ActorRole.RISK_SERVICE, ActorRole.ADMIN},
        "SUBMIT_TELEMETRY": {ActorRole.MERCHANT, ActorRole.RISK_SERVICE, ActorRole.ADMIN},
        "VIEW_FEATURE_CONTRACT": {ActorRole.RISK_ANALYST, ActorRole.RISK_SERVICE, ActorRole.ADMIN},
        "CREATE_CHECKOUT": {ActorRole.MERCHANT, ActorRole.RISK_SERVICE, ActorRole.ADMIN},
        "VERIFY_CHECKOUT": {ActorRole.MERCHANT, ActorRole.RISK_SERVICE, ActorRole.ADMIN},
    }

    async def healthcheck(self) -> None:
        return None

    async def authorize(self, request: AuthorizationRequest) -> AuthorizationDecision:
        allowed_roles = self._roles_by_action.get(request.action, set())
        if request.principal.role not in allowed_roles:
            return self._deny("ROLE_NOT_ALLOWED")
        if request.principal.role == ActorRole.MERCHANT and (
            not request.principal.merchant_id
            or request.merchant_id != request.principal.merchant_id
        ):
            return self._deny("MERCHANT_RESOURCE_MISMATCH")
        return AuthorizationDecision(
            allowed=True,
            reasons=["AUTHORIZED"],
            policy_version=self.version,
        )

    def _deny(self, reason: str) -> AuthorizationDecision:
        return AuthorizationDecision(
            allowed=False,
            reasons=[reason],
            policy_version=self.version,
        )


class OpaAuthorizer:
    def __init__(self, base_url: str, timeout_seconds: float = 2.0) -> None:
        self._health_url = f"{base_url.rstrip('/')}/health"
        self._url = f"{base_url.rstrip('/')}/v1/data/razortrust/authz/decision"
        self._timeout = timeout_seconds

    async def healthcheck(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(self._health_url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AuthorizationUnavailable("OPA authorization endpoint is unavailable") from exc

    async def authorize(self, request: AuthorizationRequest) -> AuthorizationDecision:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._url, json={"input": request.model_dump(mode="json")}
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("OPA response must be an object")
                result = payload.get("result")
            return AuthorizationDecision.model_validate(result)
        except (httpx.HTTPError, ValueError) as exc:
            raise AuthorizationUnavailable("OPA authorization endpoint is unavailable") from exc
