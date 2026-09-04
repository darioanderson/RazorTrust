from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experimental.public_fraud.public_fraud import (
    ULB_COLUMNS,
    _chronological_split,
    _write_precision_recall_plot,
    business_cost,
    detector_metric,
    load_ulb_credit_card,
    optimize_business_threshold,
)


def test_ulb_loader_validates_schema_and_hashes_bytes(tmp_path) -> None:
    rows = []
    for label in (0, 1):
        row = {column: 0.0 for column in ULB_COLUMNS}
        row["Amount"] = 10.0
        row["Class"] = label
        rows.append(row)
    path = tmp_path / "creditcard.csv"
    pd.DataFrame(rows, columns=ULB_COLUMNS).to_csv(path, index=False)

    frame, digest = load_ulb_credit_card(path)

    assert len(frame) == 2
    assert len(digest) == 64


def test_ulb_loader_rejects_non_finite_features(tmp_path) -> None:
    row = {column: 0.0 for column in ULB_COLUMNS}
    row["Amount"] = 10.0
    row["V1"] = np.nan
    path = tmp_path / "bad.csv"
    pd.DataFrame([{**row, "Class": 0}, {**row, "Class": 1}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-finite"):
        load_ulb_credit_card(path)


def test_business_threshold_uses_observed_amount_losses() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.01, 0.40, 0.30, 0.90])
    amounts = np.asarray([100.0, 100.0, 1000.0, 2000.0])

    selected = optimize_business_threshold(labels, probabilities, amounts)
    measured = business_cost("test", labels, probabilities, amounts, selected.threshold)

    assert selected.threshold == measured.threshold
    assert measured.total_cost == measured.fraud_loss + measured.legitimate_delay_cost


def test_detector_metrics_and_chronological_split(tmp_path, monkeypatch) -> None:
    labels = np.asarray([0, 1] * 5)
    scores = np.linspace(0.0, 1.0, len(labels))
    metric = detector_metric("fixture", labels, scores, threshold=0.5)

    assert metric.name == "fixture"
    assert 0 <= metric.precision <= 1
    assert 0 <= metric.recall <= 1
    frame = pd.DataFrame({"Class": labels})
    train, validation, test = _chronological_split(frame)
    assert (len(train), len(validation), len(test)) == (6, 2, 2)

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    plot_path = tmp_path / "pr.png"
    _write_precision_recall_plot(plot_path, labels, {"fixture": scores})
    assert plot_path.stat().st_size > 0


def test_chronological_split_rejects_a_partition_without_both_classes() -> None:
    with pytest.raises(ValueError, match="partitions"):
        _chronological_split(pd.DataFrame({"Class": [0] * 9 + [1]}))
