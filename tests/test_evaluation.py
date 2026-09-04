from __future__ import annotations

import pytest

from razortrust.ml.dataset import build_training_frame
from razortrust.ml.evaluation import evaluate_sealed_test, write_evaluation_report
from razortrust.ml.modeling import train_model_bundle
from razortrust.ml.splits import create_split_manifest
from razortrust.synthetic import generate_dataset


@pytest.fixture(scope="module")
def evaluated_model():
    frame = build_training_frame(
        *generate_dataset(seed=53, merchants_per_family=12, transactions_per_merchant=12)
    )
    split = create_split_manifest(frame, "d" * 64, seed=53)
    result = train_model_bundle(frame, split, seed=53, n_estimators=30)
    return evaluate_sealed_test(frame, result.bundle, split)


def test_sealed_evaluation_has_per_class_and_cost_metrics(evaluated_model) -> None:
    assert evaluated_model.row_count > 0
    assert len(evaluated_model.classes) == 3
    assert evaluated_model.expected_cost_units >= 0
    assert 0 <= evaluated_model.false_release_rate <= 1
    assert len(evaluated_model.content_sha256) == 64
    assert 0 <= evaluated_model.top_label_ece <= 1
    assert {interval.metric for interval in evaluated_model.confidence_intervals} == {
        "expected_cost_units",
        "false_release_rate",
        "escalation_recall",
    }
    assert len(evaluated_model.baselines) == 6
    assert len(evaluated_model.cost_sensitivity) == 3


def test_sealed_evaluation_is_immutable(evaluated_model, tmp_path) -> None:
    path = tmp_path / "sealed.json"
    write_evaluation_report(path, evaluated_model)
    write_evaluation_report(path, evaluated_model)
    changed = evaluated_model.model_copy(update={"row_count": evaluated_model.row_count + 1})
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_evaluation_report(path, changed)
