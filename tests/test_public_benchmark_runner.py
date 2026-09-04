from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_public_fraud_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_public_fraud_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openml_loader_retries_then_succeeds(monkeypatch) -> None:
    module = _load_module()
    calls = 0
    clears = 0

    def fake_fetch_openml(**kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("incomplete transfer")
        return SimpleNamespace(frame=pd.DataFrame({"V1": [0.1, 0.2], "Class": [0, 1]}))

    def fake_clear() -> Path:
        nonlocal clears
        clears += 1
        return Path("fake-openml-cache")

    monkeypatch.setattr(module, "fetch_openml", fake_fetch_openml)
    monkeypatch.setattr(module, "_clear_openml_cache", fake_clear)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    frame, source = module._load_openml_frame(1597, attempts=3, retry_delay_seconds=0)

    assert calls == 3
    assert clears == 2
    assert source == "openml:1597"
    assert list(frame.columns) == ["V1", "Class"]


def test_openml_loader_fails_with_local_csv_guidance(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "fetch_openml",
        lambda **kwargs: (_ for _ in ()).throw(OSError("network interrupted")),
    )
    monkeypatch.setattr(module, "_clear_openml_cache", lambda: Path("cache"))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match=r"--csv <path-to-creditcard\.csv>"):
        module._load_openml_frame(1597, attempts=2, retry_delay_seconds=0)


def test_csv_loader_is_explicit_and_does_not_use_openml(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    csv_path = tmp_path / "creditcard.csv"
    pd.DataFrame({"V1": [0.1, 0.2], "Class": [0, 1]}).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        module,
        "fetch_openml",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("OpenML should not be called")),
    )

    frame, source = module._load_frame(
        csv_path,
        None,
        openml_attempts=5,
        openml_retry_delay_seconds=5.0,
    )

    assert source == f"csv:{csv_path.resolve()}"
    assert frame.shape == (2, 2)
