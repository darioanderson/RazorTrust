from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from razortrust.security import generate_release_keypair, sign_manifest

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.e2e.yml"
BASE_URL = "http://localhost:18000"


def request(
    method: str,
    path: str,
    *,
    token: str | None = None,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    expected: set[int] | frozenset[int] = frozenset({200}),
) -> tuple[int, bytes, dict[str, str]]:
    request_headers = dict(headers or {})
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        request_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=body, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            status, payload = response.status, response.read()
            response_headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        status, payload, response_headers = exc.code, exc.read(), dict(exc.headers.items())
    if status not in expected:
        raise AssertionError(f"{method} {path}: expected {expected}, got {status}: {payload!r}")
    return status, payload, response_headers


def json_request(
    method: str,
    path: str,
    payload: object,
    *,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    expected: set[int] | frozenset[int] = frozenset({200}),
) -> dict[str, object]:
    _, raw, _ = request(
        method,
        path,
        body=json.dumps(payload).encode(),
        token=token,
        headers=headers,
        expected=expected,
    )
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise AssertionError(f"expected object from {path}")
    return decoded


def wait_ready(path: str = "/health/live", port: int = 18000, expected_status: int = 200) -> None:
    global BASE_URL
    prior = BASE_URL
    BASE_URL = f"http://localhost:{port}"
    try:
        for _ in range(90):
            try:
                request("GET", path, expected={expected_status})
                return
            except (AssertionError, OSError, urllib.error.URLError):
                time.sleep(1)
        raise TimeoutError(f"service on port {port} did not start")
    finally:
        BASE_URL = prior


def evaluation_payload(as_of: datetime) -> dict[str, object]:
    def transaction(identifier: str, hours: int, amount: int, device: str, geo: str) -> dict:
        return {
            "transaction_id": identifier,
            "merchant_id": "merchant_001",
            "timestamp": (as_of - timedelta(hours=hours)).isoformat(),
            "amount": amount,
            "device_fingerprint": device,
            "customer_geo": geo,
            "auth_status": "APPROVED",
        }

    return {
        "baseline": {
            "volume_mean": 4,
            "volume_std": 2,
            "gmv_mean": 400,
            "gmv_std": 100,
            "ticket_size_mean": 100,
            "ticket_size_std": 20,
            "refund_rate_mean": 0.1,
            "refund_rate_std": 0.05,
            "chargeback_rate_mean": 0.02,
            "chargeback_rate_std": 0.01,
            "known_devices": ["device_1"],
            "known_geos": ["IN"],
            "amount_bin_edges": [0, 100, 200],
            "amount_bin_probabilities": [0.5, 0.5],
        },
        "transactions": [
            transaction("txn_1", 10, 50, "device_1", "IN"),
            transaction("txn_2", 8, 80, "device_1", "IN"),
            transaction("txn_3", 4, 150, "device_3", "US"),
            transaction("txn_4", 1, 120, "device_2", "IN"),
        ],
    }


def main() -> None:
    merchant_token, analyst_token, service_token = (secrets.token_urlsafe(24) for _ in range(3))
    private_key, public_key = generate_release_keypair()
    principals = {
        hashlib.sha256(merchant_token.encode()).hexdigest(): {
            "id": "merchant-user",
            "role": "MERCHANT",
            "merchant_id": "merchant_001",
        },
        hashlib.sha256(analyst_token.encode()).hexdigest(): {
            "id": "analyst-user",
            "role": "RISK_ANALYST",
        },
        hashlib.sha256(service_token.encode()).hexdigest(): {
            "id": "risk-service",
            "role": "RISK_SERVICE",
        },
    }
    environment = os.environ.copy()
    environment.update(
        {
            "E2E_POSTGRES_PASSWORD": secrets.token_urlsafe(24),
            "E2E_API_PRINCIPALS": json.dumps(principals, separators=(",", ":")),
            "E2E_EVIDENCE_PUBLIC_KEYS": json.dumps({"e2e": public_key}),
            "E2E_RAZORPAY_KEY_SECRET": secrets.token_urlsafe(24),
            "E2E_WEBHOOK_SECRET": secrets.token_urlsafe(24),
        }
    )
    compose = ["docker", "compose", "-f", str(COMPOSE)]
    subprocess.run([*compose, "down", "--volumes", "--remove-orphans"], cwd=ROOT, env=environment)
    try:
        subprocess.run(
            [*compose, "up", "--build", "-d", "--wait"], cwd=ROOT, env=environment, check=True
        )
        wait_ready()
        request("GET", "/health/ready", expected={200})
        request("GET", "/v1/holds", expected={401})

        webhook = {
            "entity": "event",
            "account_id": "acc_e2e",
            "event": "payment.captured",
            "contains": [],
            "created_at": int(time.time()),
            "payload": {},
        }
        webhook_body = json.dumps(webhook, separators=(",", ":")).encode()
        signature = hmac.new(
            environment["E2E_WEBHOOK_SECRET"].encode(), webhook_body, hashlib.sha256
        ).hexdigest()
        webhook_headers = {"X-Razorpay-Signature": signature, "x-razorpay-event-id": "evt_e2e_001"}
        first = json.loads(
            request(
                "POST",
                "/v1/integrations/razorpay/webhook",
                body=webhook_body,
                headers=webhook_headers,
            )[1]
        )
        second = json.loads(
            request(
                "POST",
                "/v1/integrations/razorpay/webhook",
                body=webhook_body,
                headers=webhook_headers,
            )[1]
        )
        assert first["duplicate"] is False and second["duplicate"] is True
        bad_headers = {"X-Razorpay-Signature": "0" * 64, "x-razorpay-event-id": "evt_e2e_bad"}
        request(
            "POST",
            "/v1/integrations/razorpay/webhook",
            body=webhook_body,
            headers=bad_headers,
            expected={401},
        )
        json_request("POST", "/v1/integrations/razorpay/process-pending", {}, token=service_token)
        feature = request(
            "GET",
            "/v1/integrations/razorpay/accounts/acc_e2e/feature-contract",
            token=service_token,
            expected={200},
        )
        feature_contract = json.loads(feature[1])
        assert feature_contract["production_action_eligible"] is False
        assert feature_contract["blockers"]

        as_of = datetime.now(UTC).replace(microsecond=0)
        hold_payload = {
            "request_id": str(uuid4()),
            "merchant_id": "merchant_001",
            "source_event_id": "settlement_e2e_001",
            "triggered_at": as_of.isoformat(),
            "reason_code": "CAMPAIGN",
        }
        created = json_request(
            "POST", "/v1/holds", hold_payload, token=merchant_token, expected={201}
        )
        hold_id = str(created["hold_id"])
        request("GET", f"/v1/holds/{hold_id}/audit", token=merchant_token, expected={403})
        decision = json_request(
            "POST", f"/v1/holds/{hold_id}/evaluate", evaluation_payload(as_of), token=analyst_token
        )
        if decision["decision"] == "EVIDENCE_NEEDED":
            observed = as_of - timedelta(hours=1)
            digest = hashlib.sha256(b"e2e-campaign-attestation").hexdigest()
            attestation = {
                "schema_version": "1.0",
                "content_sha256": digest,
                "evidence_type": "CAMPAIGN",
                "subject_merchant_id": "merchant_001",
                "evidence_observed_at": observed.isoformat(),
            }
            evidence = {
                "request_id": str(uuid4()),
                "evidence_type": "CAMPAIGN",
                "submitted_at": as_of.isoformat(),
                "evidence_observed_at": observed.isoformat(),
                "content_sha256": digest,
                "metadata": {
                    "attestation_key_id": "e2e",
                    "attestation_signature": sign_manifest(attestation, private_key),
                    "subject_merchant_id": "merchant_001",
                },
            }
            json_request("POST", f"/v1/holds/{hold_id}/evidence", evidence, token=merchant_token)
            decision = json_request("POST", f"/v1/holds/{hold_id}/rescore", {}, token=analyst_token)

        if decision["decision"] != "RELEASE":
            review = {
                "request_id": str(uuid4()),
                "action": "OVERRIDE_AI",
                "authorized_decision": "RELEASE",
                "reason_code": "E2E_HUMAN_AUTHORIZATION",
                "rationale": (
                    "Named analyst authorizes this exact settlement after reviewing "
                    "the signed evidence and audit trail."
                ),
                "decided_at": datetime.now(UTC).isoformat(),
                "authorized_item": "settlement_e2e_001",
                "transaction_identity": "settlement_e2e_001",
            }
        else:
            review = {
                "request_id": str(uuid4()),
                "action": "APPROVE_RELEASE",
                "reason_code": "E2E_HUMAN_AUTHORIZATION",
                "rationale": (
                    "Named analyst authorizes this exact settlement after reviewing "
                    "the signed evidence and audit trail."
                ),
                "decided_at": datetime.now(UTC).isoformat(),
                "authorized_item": "settlement_e2e_001",
                "transaction_identity": "settlement_e2e_001",
            }
        json_request("POST", f"/v1/holds/{hold_id}/analyst-outcome", review, token=analyst_token)
        audit_before = request("GET", f"/v1/holds/{hold_id}/audit", token=analyst_token)[1]
        pdf = request("GET", f"/v1/holds/{hold_id}/dossier.pdf", token=analyst_token)[1]
        assert pdf.startswith(b"%PDF")

        subprocess.run([*compose, "restart", "api"], cwd=ROOT, env=environment, check=True)
        wait_ready()
        request("GET", "/health/ready", expected={200})
        audit_after = request("GET", f"/v1/holds/{hold_id}/audit", token=analyst_token)[1]
        assert audit_after == audit_before

        subprocess.run([*compose, "stop", "opa"], cwd=ROOT, env=environment, check=True)
        request("GET", "/health/ready", expected={503})
        request("GET", f"/v1/holds/{hold_id}", token=analyst_token, expected={503})
        subprocess.run([*compose, "start", "opa"], cwd=ROOT, env=environment, check=True)
        wait_ready("/health/ready")
        request("GET", "/health/ready", expected={200})

        prior = BASE_URL
        globals()["BASE_URL"] = "http://localhost:18001"
        try:
            wait_ready(port=18001)
            request("GET", "/health/ready", expected={503})
        finally:
            globals()["BASE_URL"] = prior

        globals()["BASE_URL"] = "http://localhost:18002"
        try:
            wait_ready(path="/health/ready", port=18002)
            _, human_only_raw, _ = request("GET", "/health/ready", expected={200})
            human_only_ready = json.loads(human_only_raw)
            assert human_only_ready["decision_mode"] == "human_only"
            assert human_only_ready["risk_runtime"] == "human-only@1"
        finally:
            globals()["BASE_URL"] = prior
        print(
            "Docker E2E passed: persistence, recovery, deduplication, signatures, "
            "RBAC, OPA/model outages, human-only readiness, audit and PDF"
        )
    finally:
        subprocess.run(
            [*compose, "down", "--volumes", "--remove-orphans"], cwd=ROOT, env=environment
        )


if __name__ == "__main__":
    main()
