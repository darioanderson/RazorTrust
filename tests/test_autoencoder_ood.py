from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from razortrust.ml.autoencoder_ood import (
    COMPACT_ARCHITECTURE,
    AutoencoderOODScorer,
    calibrate_upper_tail,
)


def test_dual_view_autoencoder_scores_shifted_points_as_more_novel() -> None:
    rng = np.random.default_rng(20260904)
    columns = [f"f{i}" for i in range(13)]
    legitimate = pd.DataFrame(rng.normal(0, 1, size=(100, 13)), columns=columns)
    groups = pd.Series([f"merchant-{index}" for index in range(len(legitimate))])
    shifted = pd.DataFrame(rng.normal(4.5, 0.7, size=(20, 13)), columns=columns)

    scorer = AutoencoderOODScorer(
        architecture=COMPACT_ARCHITECTURE,
        seed=7,
        epochs=35,
        patience=8,
    ).fit(legitimate, groups)
    normal_scores = scorer.score_details(legitimate.iloc[:20])
    shifted_scores = scorer.score_details(shifted)

    assert np.isfinite(normal_scores.reconstruction_error).all()
    assert np.isfinite(normal_scores.latent_mahalanobis_sq).all()
    assert float(np.median(shifted_scores.reconstruction_error)) > float(
        np.median(normal_scores.reconstruction_error)
    )
    assert float(np.median(shifted_scores.latent_mahalanobis_sq)) > float(
        np.median(normal_scores.latent_mahalanobis_sq)
    )


def test_upper_tail_calibration_is_finite_sample_and_monotonic() -> None:
    scores = np.linspace(0.0, 1.0, 100)
    strict = calibrate_upper_tail(scores, target_false_alarm_rate=0.01)
    loose = calibrate_upper_tail(scores, target_false_alarm_rate=0.05)
    assert strict.threshold >= loose.threshold
    assert strict.calibration_size == 100


def test_autoencoder_ood_rejects_feature_schema_changes() -> None:
    rng = np.random.default_rng(9)
    legitimate = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("abcd"))
    groups = pd.Series([f"m-{i}" for i in range(50)])
    scorer = AutoencoderOODScorer(seed=9, epochs=10, patience=3).fit(legitimate, groups)
    with pytest.raises(ValueError, match="feature columns"):
        scorer.score_details(legitimate[["b", "a", "c", "d"]])
