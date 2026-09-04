from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backup_restore_drill.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("backup_restore_drill", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restore_drill_refuses_live_database(tmp_path: Path) -> None:
    module = _load_module()
    compose = tmp_path / "compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be the live"):
        module.run_drill(
            compose_file=compose,
            backup_path=tmp_path / "backup.dump",
            restore_database="razortrust",
        )


def test_restore_drill_requires_real_restore_and_schema_checks(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    compose = tmp_path / "compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")
    backup = tmp_path / "backup.dump"
    calls: list[tuple[str, ...]] = []

    def fake_run(*args: str, capture: bool = False) -> str:
        calls.append(args)
        if args[:7] == (
            "docker",
            "compose",
            "-f",
            str(compose),
            "ps",
            "-q",
            "postgres",
        ):
            return "postgres-container"
        if args[:2] == ("docker", "cp"):
            backup.write_bytes(b"custom-format-backup")
        if args and args[-1].startswith("--command=SELECT count(*)"):
            return "17"
        if args and args[-1].startswith("--command=SELECT version_num"):
            return "20260903_0013"
        return ""

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    result = module.run_drill(compose_file=compose, backup_path=backup)

    assert result["restore_verified"] is True
    assert result["restored_public_tables"] == 17
    assert result["alembic_version"] == "20260903_0013"
    assert any("pg_restore" in call for call in calls)
    assert any("createdb" in call for call in calls)
