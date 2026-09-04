from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .security_middleware import parse_trusted_hosts


def _env_or_file(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    file_name = os.getenv(f"{name}_FILE")
    if value and file_name:
        raise ValueError(f"configure only one of {name} or {name}_FILE")
    if file_name:
        path = Path(file_name)
        if not path.is_file():
            raise ValueError(f"{name}_FILE does not exist")
        value = path.read_text(encoding="utf-8").strip()
    return value if value not in {None, ""} else default


def _strict_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be 'true' or 'false'")
    return normalized == "true"


@dataclass(frozen=True, slots=True)
class Settings:
    environment: str = "development"
    policy_mode: str = "opa"
    opa_url: str = "http://localhost:8181"
    ledger_path: str = "var/audit.jsonl"
    database_url: str | None = None
    model_release_path: str | None = None
    model_public_key: str | None = None
    decision_mode: str = "model"
    authorization_required: bool = False
    authorization_mode: str = "local"
    api_principals: dict[str, dict[str, str]] = field(default_factory=dict)
    trusted_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "test", "testserver")
    rate_limit_per_minute: int = 120
    security_headers_enabled: bool = True
    evidence_attestation_keys: dict[str, str] = field(default_factory=dict)
    razorpay_enabled: bool = False
    razorpay_mode: str = "test"
    razorpay_key_id: str | None = None
    razorpay_key_secret: str | None = None
    razorpay_webhook_secret: str | None = None
    razorpay_base_url: str = "https://api.razorpay.com/v1"
    automation_enabled: bool = False
    automation_interval_seconds: int = 60

    @classmethod
    def from_env(cls) -> Settings:
        policy_mode = os.getenv("RAZORTRUST_POLICY_MODE", "opa").lower()
        if policy_mode not in {"opa", "local"}:
            raise ValueError("RAZORTRUST_POLICY_MODE must be 'opa' or 'local'")
        authorization_mode = os.getenv("RAZORTRUST_AUTHORIZATION_MODE", "local").lower()
        if authorization_mode not in {"opa", "local"}:
            raise ValueError("RAZORTRUST_AUTHORIZATION_MODE must be 'opa' or 'local'")
        environment = os.getenv("RAZORTRUST_ENV", "development")
        decision_mode = os.getenv("RAZORTRUST_DECISION_MODE", "model").lower()
        if decision_mode not in {"model", "human_only"}:
            raise ValueError("RAZORTRUST_DECISION_MODE must be 'model' or 'human_only'")
        razorpay_mode = os.getenv("RAZORTRUST_RAZORPAY_MODE", "test").lower()
        if razorpay_mode not in {"test", "live"}:
            raise ValueError("RAZORTRUST_RAZORPAY_MODE must be 'test' or 'live'")
        principals = json.loads(_env_or_file("RAZORTRUST_API_PRINCIPALS", "{}") or "{}")
        if not isinstance(principals, dict):
            raise ValueError("RAZORTRUST_API_PRINCIPALS must be a JSON object")
        evidence_keys = json.loads(_env_or_file("RAZORTRUST_EVIDENCE_PUBLIC_KEYS", "{}") or "{}")
        if not isinstance(evidence_keys, dict):
            raise ValueError("RAZORTRUST_EVIDENCE_PUBLIC_KEYS must be a JSON object")
        return cls(
            environment=environment,
            policy_mode=policy_mode,
            opa_url=os.getenv("RAZORTRUST_OPA_URL", "http://localhost:8181"),
            ledger_path=os.getenv("RAZORTRUST_LEDGER_PATH", "var/audit.jsonl"),
            database_url=_env_or_file("RAZORTRUST_DATABASE_URL"),
            model_release_path=os.getenv("RAZORTRUST_MODEL_RELEASE_PATH"),
            model_public_key=os.getenv("RAZORTRUST_MODEL_PUBLIC_KEY"),
            decision_mode=decision_mode,
            authorization_required=_strict_bool(
                "RAZORTRUST_AUTHORIZATION_REQUIRED", environment != "development"
            ),
            authorization_mode=authorization_mode,
            api_principals=principals,
            trusted_hosts=parse_trusted_hosts(
                os.getenv(
                    "RAZORTRUST_TRUSTED_HOSTS",
                    "localhost,127.0.0.1,test,testserver" if environment == "development" else "",
                )
            ),
            rate_limit_per_minute=int(os.getenv("RAZORTRUST_RATE_LIMIT_PER_MINUTE", "120")),
            security_headers_enabled=_strict_bool("RAZORTRUST_SECURITY_HEADERS_ENABLED", True),
            evidence_attestation_keys=evidence_keys,
            razorpay_enabled=_strict_bool("RAZORTRUST_RAZORPAY_ENABLED", False),
            razorpay_mode=razorpay_mode,
            razorpay_key_id=_env_or_file("RAZORTRUST_RAZORPAY_KEY_ID"),
            razorpay_key_secret=_env_or_file("RAZORTRUST_RAZORPAY_KEY_SECRET"),
            razorpay_webhook_secret=_env_or_file("RAZORTRUST_RAZORPAY_WEBHOOK_SECRET"),
            razorpay_base_url=os.getenv(
                "RAZORTRUST_RAZORPAY_BASE_URL", "https://api.razorpay.com/v1"
            ),
            automation_enabled=_strict_bool("RAZORTRUST_AUTOMATION_ENABLED", False),
            automation_interval_seconds=max(
                5, int(os.getenv("RAZORTRUST_AUTOMATION_INTERVAL_SECONDS", "60"))
            ),
        )
