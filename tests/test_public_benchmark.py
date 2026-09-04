from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from razortrust.public_benchmark import (
    EXPECTED_FRAUD_ROWS,
    EXPECTED_ROWS,
    FEATURE_NAMES,
    SOURCE_COLUMNS,
    SOURCE_URL,
    PublicBenchmarkService,
    choose_cost_threshold,
    conformal_prediction_set,
)


def test_benchmark_feature_contract_excludes_label() -> None:
    assert len(FEATURE_NAMES) == 30
    assert "Class" not in FEATURE_NAMES
    assert FEATURE_NAMES[-2:] == ("Time", "Amount")


def test_full_source_schema_preserves_real_time_and_label() -> None:
    expected = tuple(
        ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    )
    assert expected == SOURCE_COLUMNS
    assert len(SOURCE_COLUMNS) == 31
    assert SOURCE_URL.startswith("https://storage.googleapis.com/")
    assert EXPECTED_ROWS == 284_807
    assert EXPECTED_FRAUD_ROWS == 492


def test_cost_threshold_prefers_lower_threshold_when_fn_is_expensive() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.01, 0.10, 0.30, 0.35, 0.60])
    threshold, cost = choose_cost_threshold(
        labels,
        probabilities,
        false_negative_cost=20.0,
        false_positive_cost=1.0,
    )
    assert 0 < threshold <= 0.35
    assert cost >= 0


def test_conformal_prediction_set_is_non_empty_for_typical_case() -> None:
    scores = np.array([0.02, 0.05, 0.10, 0.20, 0.35, 0.50])
    prediction_set, p_values = conformal_prediction_set(
        0.90,
        scores,
        alpha=0.10,
    )
    assert prediction_set
    assert set(p_values) == {"LEGITIMATE", "FRAUD"}
    assert 0 <= p_values["FRAUD"] <= 1


def test_blind_case_listing_hides_ground_truth() -> None:
    service = PublicBenchmarkService(root="/tmp/not-used")
    runtime = SimpleNamespace(
        test_y=np.array([0, 1, 0, 1, 0], dtype=np.int8),
        test_row_ids=np.array([10, 11, 12, 13, 14]),
        test_x=np.zeros((5, len(FEATURE_NAMES)), dtype=np.float32),
    )
    service._runtime = runtime

    items = service.list_cases(label="blind", limit=3)

    assert len(items) == 3
    assert all(item.source_label is None for item in items)
    assert all(item.source_label_name is None for item in items)
    assert all(item.label_revealed is False for item in items)


def test_labelled_analysis_mode_still_reveals_requested_truth() -> None:
    service = PublicBenchmarkService(root="/tmp/not-used")
    runtime = SimpleNamespace(
        test_y=np.array([0, 1, 0, 1, 0], dtype=np.int8),
        test_row_ids=np.array([10, 11, 12, 13, 14]),
        test_x=np.zeros((5, len(FEATURE_NAMES)), dtype=np.float32),
    )
    service._runtime = runtime

    items = service.list_cases(label="fraud", limit=2)

    assert [item.source_label for item in items] == [1, 1]
    assert all(item.source_label_name == "FRAUD" for item in items)
    assert all(item.label_revealed is True for item in items)



def test_execute_case_reads_held_out_truth_only_after_recommendation() -> None:
    import inspect

    source = inspect.getsource(PublicBenchmarkService.execute)

    truth_statement = "source_label = int(runtime.test_y[position])"

    assert source.count(truth_statement) == 1

    scoring_index = source.index(
        "raw_probability = float(runtime.model.predict_proba(row)[0, 1])"
    )

    calibration_index = source.index(
        "calibrated_probability = float("
    )

    anomaly_index = source.index(
        "anomaly_raw = float("
    )

    conformal_index = source.index(
        "prediction_set, p_values = conformal_prediction_set("
    )

    recommendation_index = source.index(
        'recommendation = "LOW_RISK_REVIEW"'
    )

    truth_index = source.index(truth_statement)

    assert truth_index > scoring_index
    assert truth_index > calibration_index
    assert truth_index > anomaly_index
    assert truth_index > conformal_index
    assert truth_index > recommendation_index


def test_ground_truth_stage_declares_post_recommendation_access() -> None:
    import inspect

    source = inspect.getsource(PublicBenchmarkService.execute)

    assert '"truth_access_mode": "POST_RECOMMENDATION_ONLY"' in source
    assert '"used_for_model_scoring": False' in source
    assert '"used_for_calibration_of_this_case": False' in source
    assert '"held_out_labels_used_for_recommendation": False' in source


def test_benchmark_status_accepts_v13_integrity_metadata() -> None:
    from razortrust.public_benchmark import BenchmarkStatus

    status = BenchmarkStatus.model_validate(
        {
            "status": "READY",
            "ready": True,
            "split": {
                "train_rows": 170884,
                "calibration_rows": 56961,
                "test_rows": 56962,
                "split_mode": "chronological_60_20_20",
            },
            "source_reference_url": "https://www.openml.org/d/1597",
            "source_columns": ["Time", "V1", "Amount", "Class"],
            "expected_rows": 284807,
            "expected_fraud_rows": 492,
            "calibration_cost_units": 287.0,
            "false_negative_cost": 20.0,
            "false_positive_cost": 1.0,
            "if_reference_split": "CALIBRATION_LEGITIMATE_ONLY",
            "if_reference_rows": 56904,
            "held_out_labels_used_for_recommendation": False,
            "feature_names": ["V1", "Time", "Amount"],
        }
    )

    assert status.status == "READY"
    assert status.ready is True

    assert status.split is not None
    assert status.split.train_rows == 170884
    assert status.split.calibration_rows == 56961
    assert status.split.test_rows == 56962
    assert status.split.split_mode == "chronological_60_20_20"

    assert status.expected_rows == 284807
    assert status.expected_fraud_rows == 492

    assert status.false_negative_cost == 20.0
    assert status.false_positive_cost == 1.0

    assert status.if_reference_split == "CALIBRATION_LEGITIMATE_ONLY"
    assert status.if_reference_rows == 56904

    assert status.held_out_labels_used_for_recommendation is False


def test_benchmark_status_still_forbids_unknown_metadata() -> None:
    import pytest
    from pydantic import ValidationError

    from razortrust.public_benchmark import BenchmarkStatus

    with pytest.raises(
        ValidationError,
        match="extra_forbidden",
    ):
        BenchmarkStatus.model_validate(
            {
                "status": "READY",
                "ready": True,
                "unknown_integrity_field": "must-be-rejected",
            }
        )


def test_benchmark_status_source_keeps_strict_extra_forbid() -> None:
    from pathlib import Path

    source = Path(
        "src/razortrust/public_benchmark.py"
    ).read_text(encoding="utf-8")

    assert 'ConfigDict(extra="forbid")' in source
    assert "class BenchmarkSplit(StrictModel)" in source
    assert "held_out_labels_used_for_recommendation" in source

