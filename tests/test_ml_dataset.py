from __future__ import annotations

import pytest

from razortrust.features import FEATURE_COLUMNS
from razortrust.ml.dataset import (
    FORBIDDEN_MODEL_COLUMNS,
    METADATA_COLUMNS,
    build_training_frame,
    model_matrix,
)
from razortrust.ml.splits import create_split_manifest, write_split_manifest
from razortrust.synthetic import generate_dataset


def training_frame():
    return build_training_frame(
        *generate_dataset(seed=31, merchants_per_family=3, transactions_per_merchant=12)
    )


def test_training_frame_keeps_labels_and_identity_out_of_model_features() -> None:
    frame = training_frame()
    assert tuple(frame.columns) == (*FEATURE_COLUMNS, *METADATA_COLUMNS)
    assert tuple(model_matrix(frame).columns) == FEATURE_COLUMNS
    assert "merchant_id" not in model_matrix(frame)
    assert "scenario_family" not in model_matrix(frame)
    assert not set(model_matrix(frame)) & FORBIDDEN_MODEL_COLUMNS
    assert len(frame) == 42


def test_split_manifest_is_deterministic_and_has_no_merchant_overlap(tmp_path) -> None:
    frame = training_frame()
    dataset_hash = "a" * 64
    first = create_split_manifest(frame, dataset_hash, seed=42, cv_folds=5)
    second = create_split_manifest(frame, dataset_hash, seed=42, cv_folds=5)
    assert first == second

    train = set(first.train_merchants)
    calibration = set(first.calibration_merchants)
    test = set(first.test_merchants)
    assert not train & calibration
    assert not train & test
    assert not calibration & test
    assert train | calibration | test == set(frame["merchant_id"])
    for fold in first.folds:
        assert not set(fold.train_merchants) & set(fold.validation_merchants)
        assert not set(fold.validation_merchants) & test

    path = tmp_path / "split-manifest.json"
    write_split_manifest(path, first)
    write_split_manifest(path, first)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_split_manifest(path, first.model_copy(update={"seed": 99}))


def test_split_rejects_more_folds_than_training_merchants() -> None:
    merchants, transactions, holds = generate_dataset(
        seed=1, merchants_per_family=4, transactions_per_merchant=12
    )
    frame = build_training_frame(merchants, transactions, holds)
    with pytest.raises(ValueError, match="requires at least"):
        create_split_manifest(frame, "b" * 64, cv_folds=100)
