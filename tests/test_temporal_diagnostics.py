from razortrust.ml.temporal_diagnostics import run_seed_diagnostic, summarize_seed_results
from razortrust.synthetic import generate_dataset


def test_temporal_diagnostic_is_point_in_time_and_reports_ablation() -> None:
    merchants, transactions, holds = generate_dataset(
        seed=11, merchants_per_family=2, transactions_per_merchant=16
    )
    result = run_seed_diagnostic(merchants, transactions, holds, seed=11, folds=2, windows=[6, 24])
    for window in ("6", "24"):
        audit = result["windows"][window]["audit"]
        assert audit["point_in_time_violations"] == 0
        assert audit["window_violations"] == 0
        metrics = result["windows"][window]["metrics"]
        assert "tabular" in metrics
        assert "temporal_full" in metrics
        assert "add_one::burst_score_10m" in metrics
        assert "drop_one::amount_autocorrelation" in metrics
    summary = summarize_seed_results([result], [6, 24])
    assert summary["windows"]["24"]["seed_count"] == 1
