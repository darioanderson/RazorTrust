from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from razortrust.ml.novelty import AutoencoderNoveltyScorer  # noqa: E402


def test_autoencoder_returns_continuous_percentile_uncertainty() -> None:
    rng = np.random.default_rng(42)
    legitimate = pd.DataFrame(rng.normal(size=(40, 6)), columns=list("abcdef"))
    scorer = AutoencoderNoveltyScorer(seed=42, epochs=5).fit(legitimate)
    scores = scorer.transform(legitimate.iloc[:8])
    assert np.all((scores >= 0) & (scores <= 1))
    assert len(np.unique(scores)) > 1


def test_autoencoder_rejects_schema_changes() -> None:
    rng = np.random.default_rng(3)
    legitimate = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
    scorer = AutoencoderNoveltyScorer(seed=3, epochs=2).fit(legitimate)
    with pytest.raises(ValueError, match="columns"):
        scorer.transform(legitimate[["b", "a", "c"]])
