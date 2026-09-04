from __future__ import annotations

from typing import Protocol

import httpx

from .domain import HoldDecision, PolicyCheck, PolicyResult


class PolicyUnavailable(RuntimeError):
    pass


class HoldPolicy(Protocol):
    async def healthcheck(self) -> None: ...

    async def check(self, request: PolicyCheck) -> PolicyResult: ...


class LocalHoldPolicy:
    """Local mirror of the Rego guardrails for tests and offline development."""

    version = "hold-policy-local@1"

    async def healthcheck(self) -> None:
        return None

    async def check(self, request: PolicyCheck) -> PolicyResult:
        if request.system_error:
            return self._escalate("SYSTEM_ERROR")
        if request.novelty_override:
            return self._escalate("NOVELTY_OVERRIDE")
        if request.evidence_round > 1:
            return self._escalate("INVALID_EVIDENCE_ROUND")
        if request.candidate_decision == HoldDecision.RELEASE:
            if (
                request.evidence_round == 0
                and request.probabilities.release < request.thresholds.release
            ):
                return self._escalate("RELEASE_THRESHOLD_NOT_MET")
            if request.evidence_round == 1:
                if request.evidence_assessment is None:
                    return self._escalate("EVIDENCE_ASSESSMENT_MISSING")
                if request.evidence_assessment.verdict != "SUPPORTED":
                    return self._escalate("EVIDENCE_NOT_SUPPORTED")
                if request.probabilities.escalate >= request.evidence_release_risk_cap:
                    return self._escalate("EVIDENCE_RELEASE_RISK_CAP_EXCEEDED")
            return PolicyResult(
                allowed_decision=HoldDecision.RELEASE,
                guardrail_triggered=False,
                reasons=["RELEASE_GUARDRAILS_PASSED"],
                policy_version=self.version,
            )
        if request.candidate_decision == HoldDecision.EVIDENCE_NEEDED:
            if request.evidence_round == 0:
                return PolicyResult(
                    allowed_decision=HoldDecision.EVIDENCE_NEEDED,
                    guardrail_triggered=False,
                    reasons=["ONE_EVIDENCE_ROUND_AVAILABLE"],
                    policy_version=self.version,
                )
            return self._escalate("EVIDENCE_ROUND_EXHAUSTED")
        return PolicyResult(
            allowed_decision=HoldDecision.ESCALATE,
            guardrail_triggered=False,
            reasons=["CANDIDATE_ESCALATE"],
            policy_version=self.version,
        )

    def _escalate(self, reason: str) -> PolicyResult:
        return PolicyResult(
            allowed_decision=HoldDecision.ESCALATE,
            guardrail_triggered=True,
            reasons=[reason],
            policy_version=self.version,
        )


class OpaHoldPolicy:
    def __init__(self, base_url: str, timeout_seconds: float = 2.0) -> None:
        self._health_url = f"{base_url.rstrip('/')}/health"
        self._url = f"{base_url.rstrip('/')}/v1/data/razortrust/holds/decision"
        self._timeout = timeout_seconds

    async def healthcheck(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(self._health_url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise PolicyUnavailable("OPA hold-policy endpoint is unavailable") from exc

    async def check(self, request: PolicyCheck) -> PolicyResult:
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
        except (httpx.HTTPError, ValueError) as exc:
            raise PolicyUnavailable("OPA hold-policy endpoint is unavailable") from exc
        if not isinstance(result, dict):
            raise PolicyUnavailable("OPA returned no hold-policy decision")
        try:
            return PolicyResult.model_validate(result)
        except ValueError as exc:
            raise PolicyUnavailable("OPA returned an invalid hold-policy decision") from exc
