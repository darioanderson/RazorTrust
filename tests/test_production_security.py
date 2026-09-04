from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI

from razortrust.security_middleware import ProductionSecurityMiddleware
from razortrust.settings import Settings


@pytest.mark.asyncio
async def test_security_headers_host_validation_and_rate_limit() -> None:
    app = FastAPI()

    @app.get("/ok")
    async def ok() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(
        ProductionSecurityMiddleware,
        trusted_hosts=("trusted.example",),
        requests_per_minute=1,
        hsts=True,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://trusted.example"
    ) as client:
        response = await client.get("/ok")
        assert response.status_code == 200
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "max-age=" in response.headers["strict-transport-security"]
        assert (await client.get("/ok")).status_code == 429
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://evil.example"
    ) as client:
        assert (await client.get("/ok")).status_code == 400


def test_settings_load_secrets_from_files(monkeypatch, tmp_path) -> None:
    database = tmp_path / "database-url"
    principals = tmp_path / "principals"
    database.write_text("postgresql+asyncpg://service:secret@db/service", encoding="utf-8")
    principals.write_text(
        json.dumps({"token-hash": {"id": "admin", "role": "ADMIN"}}), encoding="utf-8"
    )
    monkeypatch.setenv("RAZORTRUST_ENV", "staging")
    monkeypatch.setenv("RAZORTRUST_TRUSTED_HOSTS", "api.example")
    monkeypatch.setenv("RAZORTRUST_DATABASE_URL_FILE", str(database))
    monkeypatch.setenv("RAZORTRUST_API_PRINCIPALS_FILE", str(principals))
    settings = Settings.from_env()
    assert settings.authorization_required
    assert settings.database_url == "postgresql+asyncpg://service:secret@db/service"
    assert settings.api_principals["token-hash"]["role"] == "ADMIN"


def test_settings_parse_human_only_decision_mode(monkeypatch) -> None:
    monkeypatch.setenv("RAZORTRUST_ENV", "staging")
    monkeypatch.setenv("RAZORTRUST_DECISION_MODE", "human_only")
    monkeypatch.setenv("RAZORTRUST_TRUSTED_HOSTS", "api.example")
    settings = Settings.from_env()
    assert settings.decision_mode == "human_only"


def test_settings_reject_invalid_decision_mode(monkeypatch) -> None:
    monkeypatch.setenv("RAZORTRUST_DECISION_MODE", "automatic")
    with pytest.raises(ValueError, match="RAZORTRUST_DECISION_MODE"):
        Settings.from_env()
