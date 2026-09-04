from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from razortrust.costs import DEFAULT_COST_MATRIX
from razortrust.domain import HoldDecision
from razortrust.ml.dataset import build_training_frame, model_matrix
from razortrust.ml.explain import ExplanationFidelityError, top_contributions
from razortrust.ml.modeling import (
    LABEL_TO_INDEX,
    IsolationForestScorer,
    load_model_bundle,
    optimize_thresholds,
    save_model_bundle,
    train_model_bundle,
)
from razortrust.ml.release import (
    ReleaseVerificationError,
    load_verified_model_release,
    save_model_release,
)
from razortrust.ml.splits import create_split_manifest
from razortrust.security import generate_release_keypair
from razortrust.synthetic import TrueRiskState, generate_dataset


@pytest.fixture(scope="module")
def trained_result():
    frame = build_training_frame(
        *generate_dataset(seed=43, merchants_per_family=12, transactions_per_merchant=12)
    )
    split = create_split_manifest(frame, "c" * 64, seed=42)
    return frame, split, train_model_bundle(frame, split, seed=42, n_estimators=40)


def test_phase_one_pipeline_produces_probabilities_calibration_and_policy(
    trained_result,
) -> None:
    frame, split, result = trained_result
    assert result.report.sample_weights_applied is True
    assert result.report.split_manifest_sha256 == split.content_sha256
    assert {metric.method for metric in result.report.calibration_comparison} == {
        "uncalibrated",
        "sigmoid",
        "isotonic",
        "temperature",
    }
    assert result.report.selected_calibration_method in {
        "uncalibrated",
        "sigmoid",
        "isotonic",
        "temperature",
    }
    assert result.report.threshold_selection.false_release_rate <= 0.05
    assert result.report.threshold_selection.true_release_recall >= 0.20
    assert result.report.cost_matrix_sha256 == DEFAULT_COST_MATRIX.content_sha256
    assert result.bundle.cost_matrix_sha256 == DEFAULT_COST_MATRIX.content_sha256
    assert result.report.threshold_probability_source == "merchant_group_oof"

    test = frame[frame["merchant_id"].isin(split.test_merchants)].reset_index(drop=True)
    probabilities = result.bundle.predict_proba(model_matrix(test), test["cohort"])
    assert probabilities.shape == (len(test), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    anomaly = result.bundle.anomaly_scorer.transform(model_matrix(test), test["cohort"])
    assert np.all((anomaly >= 0) & (anomaly <= 1))
    assert len(np.unique(anomaly)) > 1


def test_training_adapts_calibration_folds_for_small_development_sets() -> None:
    frame = build_training_frame(
        *generate_dataset(seed=94, merchants_per_family=8, transactions_per_merchant=24)
    )
    split = create_split_manifest(frame, "d" * 64, seed=94)
    result = train_model_bundle(frame, split, seed=94, n_estimators=10)

    assert result.report.calibration_comparison
    assert result.policy_probabilities.shape == (len(result.policy_labels), 3)


def test_model_bundle_round_trip_preserves_predictions(trained_result, tmp_path) -> None:
    frame, split, result = trained_result
    path = tmp_path / "model.joblib"
    digest = save_model_bundle(path, result)
    assert len(digest) == 64
    loaded = load_model_bundle(path)
    test = frame[frame["merchant_id"].isin(split.test_merchants)].reset_index(drop=True)
    expected = result.bundle.predict_proba(model_matrix(test), test["cohort"])
    actual = loaded.predict_proba(model_matrix(test), test["cohort"])
    assert np.allclose(actual, expected)


def test_signed_release_verifies_before_loading_and_rejects_tampering(
    trained_result, tmp_path
) -> None:
    frame, split, result = trained_result
    private_key, public_key = generate_release_keypair()
    release_path = tmp_path / "model-release"
    save_model_release(
        release_path,
        result,
        private_key_b64=private_key,
        signer_key_id="test-key",
        split_manifest_sha256=split.content_sha256,
        training_dataset_sha256="a" * 64,
        evaluation_report_sha256="b" * 64,
        release_id="test-release",
    )
    loaded = load_verified_model_release(release_path, public_key_b64=public_key)
    test = frame[frame["merchant_id"].isin(split.test_merchants)].reset_index(drop=True)
    expected = result.bundle.predict_proba(model_matrix(test), test["cohort"])
    actual = loaded.predict_proba(model_matrix(test), test["cohort"])
    assert np.allclose(actual, expected)

    metadata = release_path / "metadata.json"
    metadata.write_bytes(metadata.read_bytes() + b" ")
    with pytest.raises(ReleaseVerificationError, match="hash mismatch"):
        load_verified_model_release(release_path, public_key_b64=public_key)


def test_isolation_forest_requires_legitimate_reference_rows() -> None:
    frame = build_training_frame(
        *generate_dataset(seed=5, merchants_per_family=1, transactions_per_merchant=12)
    )
    with pytest.raises(ValueError, match="legitimate reference"):
        IsolationForestScorer().fit(
            model_matrix(frame),
            frame["cohort"],
            frame["true_risk_state"].astype(str) == "NOT_A_REAL_STATE",
        )
    assert set(frame["true_risk_state"].astype(str)) == {
        TrueRiskState.LEGITIMATE,
        TrueRiskState.RISKY,
    }


def test_cost_threshold_search_respects_false_release_constraint() -> None:
    probabilities = np.asarray(
        [
            [0.95, 0.03, 0.02],
            [0.05, 0.90, 0.05],
            [0.02, 0.08, 0.90],
            [0.10, 0.20, 0.70],
        ]
    )
    labels = np.asarray(
        [
            LABEL_TO_INDEX[HoldDecision.RELEASE],
            LABEL_TO_INDEX[HoldDecision.EVIDENCE_NEEDED],
            LABEL_TO_INDEX[HoldDecision.ESCALATE],
            LABEL_TO_INDEX[HoldDecision.ESCALATE],
        ]
    )
    selection = optimize_thresholds(probabilities, labels)
    assert selection.false_release_rate == 0
    assert selection.expected_cost_units >= 0


def test_explanation_fidelity_rejects_an_unsupported_factor() -> None:
    names = ["gmv_delta_z", "refund_rate_delta_z", "new_device_ratio", "new_geo_ratio"]
    observed = np.asarray([4.2, 2.0, 0.8, 0.1])
    contributions = np.asarray([0.9, 0.7, 0.5, 0.01])
    reasons = top_contributions(
        names,
        observed,
        contributions,
        HoldDecision.ESCALATE,
        requested_features={"gmv_delta_z"},
    )
    assert [reason.feature for reason in reasons] == names[:3]
    assert all(reason.attribution_method == "tree_shap" for reason in reasons)
    with pytest.raises(ExplanationFidelityError, match="new_geo_ratio"):
        top_contributions(
            names,
            observed,
            contributions,
            HoldDecision.ESCALATE,
            requested_features={"new_geo_ratio"},
        )


def test_isolation_forest_reference_is_group_oof_and_raw_score_stays_unsaturated() -> None:
    rng = np.random.default_rng(20260902)
    features = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    cohorts = pd.Series(["retail"] * len(features))
    legitimate = pd.Series([True] * len(features))
    merchant_groups = pd.Series([f"merchant-{index // 4}" for index in range(len(features))])

    scorer = IsolationForestScorer(seed=77).fit(features, cohorts, legitimate, merchant_groups)

    assert scorer.version == "isolation-forest@2"
    assert scorer.reference_mode == "merchant_group_oof"
    final_model_in_sample = np.sort(-scorer.models["pooled"].score_samples(features))
    assert not np.allclose(final_model_in_sample, scorer.reference_scores["pooled"])

    probes = pd.DataFrame([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], columns=features.columns)
    details = scorer.score_details(probes, pd.Series(["retail", "retail"]))
    assert np.all((details.percentiles >= 0) & (details.percentiles <= 1))
    assert details.raw_scores[1] > details.raw_scores[0]
    assert details.tail_excesses[1] >= details.tail_excesses[0]
    assert np.all(details.reference_sizes > 0)
    assert details.scorer_version == "isolation-forest@2"
    assert details.reference_mode == "merchant_group_oof"
