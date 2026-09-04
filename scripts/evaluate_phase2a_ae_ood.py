from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from razortrust.domain import HoldDecision
from razortrust.ml.autoencoder_ood import (
    DEFAULT_ARCHITECTURES,
    AutoencoderArchitecture,
    AutoencoderOODScorer,
    calibrate_upper_tail,
)
from razortrust.ml.dataset import build_training_frame, model_matrix
from razortrust.ml.modeling import IsolationForestScorer
from razortrust.ml.runtime import TrainedRiskModel
from razortrust.ml.splits import create_split_manifest, write_split_manifest
from razortrust.synthetic import (
    LEGITIMATE_FAMILIES,
    TrueRiskState,
    generate_dataset,
    write_dataset,
)

ALPHAS = (0.005, 0.01, 0.02, 0.05)
CONSENSUS_RULES = ("AE_ANY", "AE_BOTH", "TWO_OF_THREE")
SCOPE = "SYNTHETIC_RESEARCH_ONLY_PHASE2A_NO_PROMOTION"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2A dual-view AE + latent OOD research")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--merchants-per-family", type=int, default=60)
    parser.add_argument("--transactions-per-merchant", type=int, default=80)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    champion_release = ROOT / "artifacts" / "tier0" / "model-release"
    champion_public_key_path = ROOT / "artifacts" / "tier0" / "model-release-public-key.txt"
    for path in (champion_release, champion_public_key_path):
        if not path.exists():
            raise FileNotFoundError(f"required champion artifact missing: {path}")

    manifest = write_dataset(
        output / "dataset",
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    merchants, transactions, holds = generate_dataset(
        seed=args.seed,
        merchants_per_family=args.merchants_per_family,
        transactions_per_merchant=args.transactions_per_merchant,
    )
    frame = build_training_frame(merchants, transactions, holds)
    split = create_split_manifest(frame, manifest.content_sha256, seed=args.seed)
    write_split_manifest(output / "split_manifest.json", split)

    train = _partition(frame, split.train_merchants)
    calibration = _partition(frame, split.calibration_merchants)
    test = _partition(frame, split.test_merchants)
    train_legit = train[
        train["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE
    ].reset_index(drop=True)
    cal_legit_mask = (
        calibration["true_risk_state"].astype(str).to_numpy() == TrueRiskState.LEGITIMATE
    )
    cal_risk_mask = ~cal_legit_mask
    test_legit_mask = test["true_risk_state"].astype(str).to_numpy() == TrueRiskState.LEGITIMATE
    test_risk_mask = ~test_legit_mask

    if2 = IsolationForestScorer(seed=args.seed).fit(
        model_matrix(train),
        train["cohort"],
        train["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE,
        train["merchant_id"].astype(str),
    )
    if getattr(if2, "version", "") != "isolation-forest@2":
        raise RuntimeError("Phase 2A requires isolation-forest@2 research semantics")
    cal_if = if2.score_details(model_matrix(calibration), calibration["cohort"])
    test_if = if2.score_details(model_matrix(test), test["cohort"])

    champion = TrainedRiskModel.load_release(
        champion_release,
        public_key_b64=champion_public_key_path.read_text(encoding="utf-8").strip(),
    )
    test_probabilities = champion.bundle.predict_proba(model_matrix(test), test["cohort"])
    test_base_actions = _threshold_actions(
        test_probabilities, champion.thresholds.release, champion.thresholds.escalate
    )

    architecture_runs: list[dict[str, Any]] = []
    fitted_models: dict[str, AutoencoderOODScorer] = {}
    for index, architecture in enumerate(DEFAULT_ARCHITECTURES):
        scorer = AutoencoderOODScorer(
            architecture=architecture,
            seed=args.seed + 100 + index,
            epochs=120,
            patience=15,
        ).fit(model_matrix(train_legit), train_legit["merchant_id"].astype(str))
        fitted_models[architecture.name] = scorer
        cal_ae = scorer.score_details(model_matrix(calibration))
        test_ae = scorer.score_details(model_matrix(test))
        candidates = _candidate_grid(
            architecture=architecture,
            cal_ae=cal_ae,
            cal_if_raw=cal_if.raw_scores,
            legitimate_mask=cal_legit_mask,
            risky_mask=cal_risk_mask,
        )
        selected = _select_candidate(candidates)
        test_metrics, test_signal = _evaluate_selected(
            selected,
            test_ae=test_ae,
            test_if_raw=test_if.raw_scores,
            legitimate_mask=test_legit_mask,
            risky_mask=test_risk_mask,
            base_actions=test_base_actions,
        )
        architecture_runs.append(
            {
                "architecture": architecture.name,
                "best_epoch": scorer.best_epoch,
                "best_validation_loss": scorer.best_validation_loss,
                "selected_calibration_candidate": selected,
                "test_metrics": test_metrics,
            }
        )

    selected_architecture_run = max(
        architecture_runs,
        key=lambda row: (
            float(row["selected_calibration_candidate"]["selection_utility"]),
            float(row["selected_calibration_candidate"]["risk_signal_rate"]),
            -float(row["selected_calibration_candidate"]["legitimate_signal_rate"]),
        ),
    )
    # Both architecture and thresholds are selected only from the calibration partition.
    # Test metrics are report-only and cannot influence the selected configuration.
    selected_arch_name = selected_architecture_run["architecture"]
    selected_model = fitted_models[selected_arch_name]
    selected_calibration = selected_architecture_run["selected_calibration_candidate"]
    selected_test_ae = selected_model.score_details(model_matrix(test))
    selected_test_metrics, selected_test_signal = _evaluate_selected(
        selected_calibration,
        test_ae=selected_test_ae,
        test_if_raw=test_if.raw_scores,
        legitimate_mask=test_legit_mask,
        risky_mask=test_risk_mask,
        base_actions=test_base_actions,
    )

    family_metrics = _family_metrics(
        test,
        signal=selected_test_signal,
        reconstruction=selected_test_ae.reconstruction_error,
        latent=selected_test_ae.latent_mahalanobis_sq,
        if_raw=test_if.raw_scores,
    )
    benign_holdout = _leave_one_legitimate_family_out(
        frame=frame,
        architecture=_architecture_by_name(selected_arch_name),
        alpha=float(selected_calibration["target_false_alarm_rate"]),
        rule=str(selected_calibration["consensus_rule"]),
        seed=args.seed + 9000,
    )

    embeddings = pd.DataFrame(
        selected_test_ae.latent_vectors,
        columns=[f"latent_{i}" for i in range(selected_test_ae.latent_vectors.shape[1])],
    )
    embeddings.insert(0, "merchant_id", test["merchant_id"].astype(str).to_numpy())
    embeddings.insert(1, "scenario_family", test["scenario_family"].astype(str).to_numpy())
    embeddings.insert(2, "true_risk_state", test["true_risk_state"].astype(str).to_numpy())
    embeddings["reconstruction_error"] = selected_test_ae.reconstruction_error
    embeddings["latent_mahalanobis_sq"] = selected_test_ae.latent_mahalanobis_sq
    embeddings["if_raw_score"] = test_if.raw_scores
    embeddings["novelty_signal"] = selected_test_signal
    embeddings.to_csv(output / "latent_embeddings_for_visualization.csv", index=False)

    report: dict[str, Any] = {
        "status": "OK",
        "scope": SCOPE,
        "created_at": datetime.now(UTC).isoformat(),
        "promotion_eligible": False,
        "serving_change_authorized": False,
        "production_action_eligible": False,
        "champion_model_version": champion.version,
        "if_model_version": getattr(if2, "version", "unknown"),
        "if_reference_mode": getattr(if2, "reference_mode", "unknown"),
        "autoencoder_model_version": selected_model.version,
        "selected_architecture": selected_arch_name,
        "selection_note": "architecture candidates calibrated without test; test remains research-only because the synthetic generator has been repeatedly inspected",
        "selected_calibration": selected_calibration,
        "test_metrics": selected_test_metrics,
        "architecture_runs": architecture_runs,
        "risk_family_metrics": family_metrics,
        "leave_one_legitimate_family_out": benign_holdout,
        "latent_embedding_artifact": "latent_embeddings_for_visualization.csv",
        "interpretation_contract": {
            "reconstruction_error": "normal-only denoising reconstruction difficulty, not fraud probability",
            "latent_mahalanobis_sq": "distance from the legitimate latent distribution using Ledoit-Wolf shrinkage covariance",
            "consensus_signal": "unknown-risk safety signal; novelty is uncertainty, not guilt",
        },
    }
    report["report_sha256"] = _sha256_json(report)
    _write_json(output / "phase2a_ae_ood_report.json", report)

    summary = {
        "status": "OK",
        "scope": SCOPE,
        "promotion_eligible": False,
        "serving_change_authorized": False,
        "champion_model_version": champion.version,
        "autoencoder_model_version": selected_model.version,
        "selected_architecture": selected_arch_name,
        "selected_target_false_alarm_rate": selected_calibration["target_false_alarm_rate"],
        "selected_consensus_rule": selected_calibration["consensus_rule"],
        "test_legitimate_signal_rate": selected_test_metrics["legitimate_signal_rate"],
        "test_risk_signal_rate": selected_test_metrics["risk_signal_rate"],
        "test_risky_release_rescue_rate": selected_test_metrics["risky_release_rescue_rate"],
        "test_legitimate_release_block_rate": selected_test_metrics[
            "legitimate_release_block_rate"
        ],
        "report_sha256": report["report_sha256"],
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))


def _candidate_grid(
    *,
    architecture: AutoencoderArchitecture,
    cal_ae: Any,
    cal_if_raw: np.ndarray,
    legitimate_mask: np.ndarray,
    risky_mask: np.ndarray,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        recon_cal = calibrate_upper_tail(
            cal_ae.reconstruction_error[legitimate_mask], target_false_alarm_rate=alpha
        )
        latent_cal = calibrate_upper_tail(
            cal_ae.latent_mahalanobis_sq[legitimate_mask], target_false_alarm_rate=alpha
        )
        if_cal = calibrate_upper_tail(cal_if_raw[legitimate_mask], target_false_alarm_rate=alpha)
        recon_signal = cal_ae.reconstruction_error >= recon_cal.threshold
        latent_signal = cal_ae.latent_mahalanobis_sq >= latent_cal.threshold
        if_signal = cal_if_raw >= if_cal.threshold
        for rule in CONSENSUS_RULES:
            signal = _consensus(rule, recon_signal, latent_signal, if_signal)
            legit_rate = float(np.mean(signal[legitimate_mask]))
            risk_rate = float(np.mean(signal[risky_mask]))
            candidates.append(
                {
                    "architecture": architecture.name,
                    "target_false_alarm_rate": alpha,
                    "consensus_rule": rule,
                    "reconstruction_threshold": recon_cal.threshold,
                    "latent_threshold": latent_cal.threshold,
                    "if_raw_threshold": if_cal.threshold,
                    "legitimate_signal_rate": round(legit_rate, 8),
                    "risk_signal_rate": round(risk_rate, 8),
                    "selection_utility": round(risk_rate - 3.0 * legit_rate, 8),
                }
            )
    return candidates


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in candidates if float(row["legitimate_signal_rate"]) <= 0.05]
    pool = eligible or candidates
    return max(
        pool,
        key=lambda row: (
            float(row["selection_utility"]),
            float(row["risk_signal_rate"]),
            -float(row["legitimate_signal_rate"]),
            1 if row["consensus_rule"] == "TWO_OF_THREE" else 0,
        ),
    )


def _evaluate_selected(
    selected: dict[str, Any],
    *,
    test_ae: Any,
    test_if_raw: np.ndarray,
    legitimate_mask: np.ndarray,
    risky_mask: np.ndarray,
    base_actions: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    recon_signal = test_ae.reconstruction_error >= float(selected["reconstruction_threshold"])
    latent_signal = test_ae.latent_mahalanobis_sq >= float(selected["latent_threshold"])
    if_signal = test_if_raw >= float(selected["if_raw_threshold"])
    signal = _consensus(str(selected["consensus_rule"]), recon_signal, latent_signal, if_signal)
    release_mask = base_actions == HoldDecision.RELEASE
    risky_release = risky_mask & release_mask
    legitimate_release = legitimate_mask & release_mask
    combined_score = np.maximum.reduce(
        [
            _rank(test_ae.reconstruction_error),
            _rank(test_ae.latent_mahalanobis_sq),
            _rank(test_if_raw),
        ]
    )
    labels = risky_mask.astype(int)
    metrics = {
        "legitimate_signal_rate": round(float(np.mean(signal[legitimate_mask])), 8),
        "risk_signal_rate": round(float(np.mean(signal[risky_mask])), 8),
        "legitimate_release_block_rate": round(
            float(np.mean(signal[legitimate_release])) if np.any(legitimate_release) else 0.0, 8
        ),
        "risky_release_rescue_rate": round(
            float(np.mean(signal[risky_release])) if np.any(risky_release) else 0.0, 8
        ),
        "risky_release_count": int(np.count_nonzero(risky_release)),
        "legitimate_release_count": int(np.count_nonzero(legitimate_release)),
        "diagnostic_auroc": round(float(roc_auc_score(labels, combined_score)), 8),
        "diagnostic_auprc": round(float(average_precision_score(labels, combined_score)), 8),
    }
    return metrics, signal


def _leave_one_legitimate_family_out(
    *,
    frame: pd.DataFrame,
    architecture: AutoencoderArchitecture,
    alpha: float,
    rule: str,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    legitimate = frame[
        frame["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE
    ].reset_index(drop=True)
    for family_index, family in enumerate(LEGITIMATE_FAMILIES):
        training = legitimate[legitimate["scenario_family"].astype(str) != family].reset_index(
            drop=True
        )
        held_out = legitimate[legitimate["scenario_family"].astype(str) == family].reset_index(
            drop=True
        )
        # Split the non-held-out legitimate merchants so thresholds are still disjoint from AE fitting.
        merchant_ids = sorted(training["merchant_id"].astype(str).unique())
        rng = np.random.default_rng(seed + family_index)
        rng.shuffle(merchant_ids)
        cut = max(20, int(round(len(merchant_ids) * 0.18)))
        calibration_ids = set(merchant_ids[:cut])
        fit = training[~training["merchant_id"].astype(str).isin(calibration_ids)].reset_index(
            drop=True
        )
        calibration = training[
            training["merchant_id"].astype(str).isin(calibration_ids)
        ].reset_index(drop=True)
        scorer = AutoencoderOODScorer(
            architecture=architecture,
            seed=seed + family_index,
            epochs=80,
            patience=10,
        ).fit(model_matrix(fit), fit["merchant_id"].astype(str))
        cal_scores = scorer.score_details(model_matrix(calibration))
        held_scores = scorer.score_details(model_matrix(held_out))
        recon_cal = calibrate_upper_tail(
            cal_scores.reconstruction_error, target_false_alarm_rate=alpha
        )
        latent_cal = calibrate_upper_tail(
            cal_scores.latent_mahalanobis_sq, target_false_alarm_rate=alpha
        )
        if2 = IsolationForestScorer(seed=seed + 500 + family_index).fit(
            model_matrix(fit),
            fit["cohort"],
            pd.Series(True, index=fit.index),
            fit["merchant_id"].astype(str),
        )
        cal_if = if2.score_details(model_matrix(calibration), calibration["cohort"])
        held_if = if2.score_details(model_matrix(held_out), held_out["cohort"])
        if_cal = calibrate_upper_tail(cal_if.raw_scores, target_false_alarm_rate=alpha)
        signal = _consensus(
            rule,
            held_scores.reconstruction_error >= recon_cal.threshold,
            held_scores.latent_mahalanobis_sq >= latent_cal.threshold,
            held_if.raw_scores >= if_cal.threshold,
        )
        results.append(
            {
                "held_out_legitimate_family": family,
                "n": len(held_out),
                "signal_rate": round(float(np.mean(signal)), 8),
                "mean_reconstruction_error": round(
                    float(np.mean(held_scores.reconstruction_error)), 8
                ),
                "mean_latent_mahalanobis_sq": round(
                    float(np.mean(held_scores.latent_mahalanobis_sq)), 8
                ),
            }
        )
    return results


def _consensus(
    rule: str, recon: np.ndarray, latent: np.ndarray, if_signal: np.ndarray
) -> np.ndarray:
    if rule == "AE_ANY":
        return recon | latent
    if rule == "AE_BOTH":
        return recon & latent
    if rule == "TWO_OF_THREE":
        return (recon.astype(int) + latent.astype(int) + if_signal.astype(int)) >= 2
    raise ValueError(f"unknown consensus rule: {rule}")


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return (order + 1) / len(values)


def _threshold_actions(
    probabilities: np.ndarray, release_threshold: float, escalate_threshold: float
) -> np.ndarray:
    actions = np.full(len(probabilities), HoldDecision.EVIDENCE_NEEDED, dtype=object)
    release = probabilities[:, 0] >= release_threshold
    escalate = (~release) & (probabilities[:, 2] >= escalate_threshold)
    actions[release] = HoldDecision.RELEASE
    actions[escalate] = HoldDecision.ESCALATE
    return actions


def _family_metrics(
    frame: pd.DataFrame,
    *,
    signal: np.ndarray,
    reconstruction: np.ndarray,
    latent: np.ndarray,
    if_raw: np.ndarray,
) -> list[dict[str, Any]]:
    families = frame["scenario_family"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for family in sorted(set(families)):
        mask = families == family
        rows.append(
            {
                "scenario_family": family,
                "true_risk_state": str(frame.loc[np.flatnonzero(mask)[0], "true_risk_state"]),
                "n": int(np.count_nonzero(mask)),
                "signal_rate": round(float(np.mean(signal[mask])), 8),
                "mean_reconstruction_error": round(float(np.mean(reconstruction[mask])), 8),
                "mean_latent_mahalanobis_sq": round(float(np.mean(latent[mask])), 8),
                "mean_if_raw_score": round(float(np.mean(if_raw[mask])), 8),
            }
        )
    return rows


def _architecture_by_name(name: str) -> AutoencoderArchitecture:
    for architecture in DEFAULT_ARCHITECTURES:
        if architecture.name == name:
            return architecture
    raise KeyError(name)


def _partition(frame: pd.DataFrame, merchants: list[str]) -> pd.DataFrame:
    result = frame[frame["merchant_id"].isin(merchants)].reset_index(drop=True)
    if result.empty:
        raise ValueError("empty split partition")
    return result


def _sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
