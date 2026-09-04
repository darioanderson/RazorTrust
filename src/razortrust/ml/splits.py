from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ..audit import canonical_json
from ..domain import StrictModel
from ..features import FEATURE_SCHEMA_VERSION


class GroupFold(StrictModel):
    fold: int
    train_merchants: list[str]
    validation_merchants: list[str]


class SplitManifest(StrictModel):
    schema_version: str = "1.0"
    seed: int
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_schema_version: str
    train_merchants: list[str]
    calibration_merchants: list[str]
    test_merchants: list[str]
    train_merchants_sha256: str
    calibration_merchants_sha256: str
    test_merchants_sha256: str
    folds: list[GroupFold]
    content_sha256: str


def create_split_manifest(
    frame: pd.DataFrame,
    dataset_manifest_sha256: str,
    *,
    seed: int = 42,
    cv_folds: int = 5,
) -> SplitManifest:
    """Create 70/15/15 merchant partitions and CV folds inside training only."""
    groups = frame["merchant_id"].astype(str)
    development_calibration_idx, test_idx = _class_complete_group_split(
        frame,
        groups,
        test_size=0.15,
        seed=seed,
    )

    remainder = frame.iloc[development_calibration_idx]
    remainder_groups = groups.iloc[development_calibration_idx]
    train_relative_idx, calibration_relative_idx = _class_complete_group_split(
        remainder,
        remainder_groups.reset_index(drop=True),
        test_size=0.15 / 0.85,
        seed=seed + 1,
    )
    train_idx = development_calibration_idx[train_relative_idx]
    calibration_idx = development_calibration_idx[calibration_relative_idx]

    train_merchants = _merchant_list(groups.iloc[train_idx])
    calibration_merchants = _merchant_list(groups.iloc[calibration_idx])
    test_merchants = _merchant_list(groups.iloc[test_idx])
    _assert_disjoint(train_merchants, calibration_merchants, test_merchants)

    unique_train_count = len(train_merchants)
    if unique_train_count < cv_folds:
        raise ValueError(
            f"cv_folds={cv_folds} requires at least {cv_folds} training merchants; "
            f"found {unique_train_count}"
        )
    training_frame = frame[frame["merchant_id"].isin(train_merchants)].reset_index(drop=True)
    training_groups = training_frame["merchant_id"].astype(str)
    splitter = GroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    folds = [
        GroupFold(
            fold=fold_number,
            train_merchants=_merchant_list(training_groups.iloc[fold_train_idx]),
            validation_merchants=_merchant_list(training_groups.iloc[fold_validation_idx]),
        )
        for fold_number, (fold_train_idx, fold_validation_idx) in enumerate(
            splitter.split(training_frame, groups=training_groups), start=1
        )
    ]
    for fold in folds:
        if set(fold.train_merchants) & set(fold.validation_merchants):
            raise RuntimeError(f"merchant leakage detected in fold {fold.fold}")
        if set(fold.validation_merchants) & set(test_merchants):
            raise RuntimeError(f"test merchant leaked into fold {fold.fold}")

    train_merchants_sha256 = _merchant_hash(train_merchants)
    calibration_merchants_sha256 = _merchant_hash(calibration_merchants)
    test_merchants_sha256 = _merchant_hash(test_merchants)
    content = {
        "schema_version": "1.0",
        "seed": seed,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "train_merchants": train_merchants,
        "calibration_merchants": calibration_merchants,
        "test_merchants": test_merchants,
        "train_merchants_sha256": train_merchants_sha256,
        "calibration_merchants_sha256": calibration_merchants_sha256,
        "test_merchants_sha256": test_merchants_sha256,
        "folds": [fold.model_dump(mode="json") for fold in folds],
    }
    return SplitManifest(
        seed=seed,
        dataset_manifest_sha256=dataset_manifest_sha256,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        train_merchants=train_merchants,
        calibration_merchants=calibration_merchants,
        test_merchants=test_merchants,
        train_merchants_sha256=train_merchants_sha256,
        calibration_merchants_sha256=calibration_merchants_sha256,
        test_merchants_sha256=test_merchants_sha256,
        folds=folds,
        content_sha256=hashlib.sha256(canonical_json(content)).hexdigest(),
    )


def write_split_manifest(path: str | Path, manifest: SplitManifest) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (manifest.model_dump_json(indent=2) + "\n").encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(f"refusing to replace split manifest: {destination}")
        return
    destination.write_bytes(content)


def _merchant_list(values: pd.Series) -> list[str]:
    return sorted(set(values.astype(str)))


def _merchant_hash(merchants: list[str]) -> str:
    return hashlib.sha256("\n".join(merchants).encode("utf-8")).hexdigest()


def _assert_disjoint(train: list[str], calibration: list[str], test: list[str]) -> None:
    train_set = set(train)
    calibration_set = set(calibration)
    test_set = set(test)
    if train_set & calibration_set or train_set & test_set or calibration_set & test_set:
        raise RuntimeError("merchant leakage detected across outer partitions")


def _class_complete_group_split(
    frame: pd.DataFrame,
    groups: pd.Series,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the first deterministic group split containing every target class on both sides."""
    labels = frame["operational_target"].astype(str).reset_index(drop=True)
    expected = set(labels)
    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=seed)
    for left_idx, right_idx in splitter.split(frame, groups=groups):
        if set(labels.iloc[left_idx]) == expected and set(labels.iloc[right_idx]) == expected:
            return left_idx, right_idx
    raise ValueError("unable to create a group-isolated split containing every operational class")
