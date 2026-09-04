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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from razortrust.costs import DEFAULT_COST_MATRIX
from razortrust.domain import HoldDecision
from razortrust.ml.autoencoder_ood import COMPACT_ARCHITECTURE, AutoencoderOODScorer
from razortrust.ml.dataset import build_training_frame, model_matrix
from razortrust.ml.modeling import LABEL_TO_INDEX, IsolationForestScorer, train_model_bundle
from razortrust.ml.phase2b_unknown import apply_evidence_only_guard
from razortrust.ml.phase2c_family_generalization import (
    CONSENSUS_RULES,
    consensus_signal,
    family_robust_upper_tail,
)
from razortrust.ml.splits import create_split_manifest, write_split_manifest
from razortrust.synthetic import (
    LEGITIMATE_FAMILIES,
    RISK_FAMILIES,
    TrueRiskState,
    generate_dataset,
    write_dataset,
)

ALPHAS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
SCOPE = "SYNTHETIC_RESEARCH_ONLY_PHASE2C_FAMILY_GENERALIZED_NO_PROMOTION"
ROUTER_VERSION = "family-generalized-novelty-router@1-research"
FRICTION_PENALTY = 10.0

# Inner operational gates used only to choose a research guard without touching the
# outer held-out family. Risk-rescue is required only when the inner policy set
# actually contains risky RELEASE candidates.
INNER_GATES = {
    "max_legitimate_release_diversion_rate": 0.20,
    "max_worst_legitimate_family_release_diversion_rate": 0.50,
    "min_risky_release_rescue_rate_when_applicable": 0.50,
    "require_non_increasing_expected_cost": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2C nested family-generalized novelty calibration benchmark"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--merchants-per-family", type=int, default=50)
    parser.add_argument("--transactions-per-merchant", type=int, default=60)
    parser.add_argument("--xgb-estimators", type=int, default=120)
    parser.add_argument("--ae-epochs", type=int, default=60)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    champion_release = ROOT / "artifacts" / "tier0" / "model-release"
    champion_public_key = ROOT / "artifacts" / "tier0" / "model-release-public-key.txt"
    if not champion_release.exists() or not champion_public_key.exists():
        raise FileNotFoundError("current signed champion artifacts are required")

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
    frame = build_training_frame(merchants, transactions, holds).reset_index(drop=True)

    families = tuple(LEGITIMATE_FAMILIES) + tuple(RISK_FAMILIES)
    outer_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for family_index, held_out_family in enumerate(families):
        held_out = frame[frame["scenario_family"].astype(str) == held_out_family].reset_index(
            drop=True
        )
        development = frame[frame["scenario_family"].astype(str) != held_out_family].reset_index(
            drop=True
        )
        if held_out.empty or development.empty:
            raise RuntimeError(f"invalid outer family fold: {held_out_family}")

        fold_hash = hashlib.sha256(
            f"{manifest.content_sha256}|{held_out_family}|phase2c".encode()
        ).hexdigest()
        fold_seed = args.seed + 2000 + family_index
        split = create_split_manifest(development, fold_hash, seed=fold_seed)
        write_split_manifest(output / f"split_{held_out_family}.json", split)

        training_result = train_model_bundle(
            development,
            split,
            seed=fold_seed,
            n_estimators=args.xgb_estimators,
        )
        bundle = training_result.bundle

        train = _partition(development, split.train_merchants)
        calibration = _partition(development, split.calibration_merchants)
        inner_policy = _partition(development, split.test_merchants)

        pseudo_families = _choose_pseudo_unseen_legitimate_families(
            train=train,
            calibration=calibration,
            held_out_family=held_out_family,
            rotation=family_index,
        )

        novelty_train_legit = train[
            (train["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE)
            & (~train["scenario_family"].astype(str).isin(pseudo_families))
        ].reset_index(drop=True)
        calibration_legit = calibration[
            calibration["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE
        ].reset_index(drop=True)
        pseudo_train_legit = train[
            (train["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE)
            & (train["scenario_family"].astype(str).isin(pseudo_families))
        ].reset_index(drop=True)
        # The pseudo families were excluded from novelty-model fitting, so their
        # train-partition rows are safe to use as a larger family-shift calibration
        # reference. Ordinary calibration legitimate rows are included as a
        # secondary in-distribution reference.
        family_calibration_legit = pd.concat(
            [calibration_legit, pseudo_train_legit], ignore_index=True
        )
        if len(novelty_train_legit) < 40 or len(family_calibration_legit) < 40:
            raise RuntimeError(
                f"insufficient novelty train/family-calibration rows in {held_out_family}"
            )

        ae = AutoencoderOODScorer(
            architecture=COMPACT_ARCHITECTURE,
            seed=fold_seed + 300,
            epochs=args.ae_epochs,
            patience=10,
        ).fit(
            model_matrix(novelty_train_legit),
            novelty_train_legit["merchant_id"].astype(str),
        )

        if_scorer = IsolationForestScorer(seed=fold_seed + 400).fit(
            model_matrix(novelty_train_legit),
            novelty_train_legit["cohort"],
            pd.Series(True, index=novelty_train_legit.index),
            novelty_train_legit["merchant_id"].astype(str),
        )
        if getattr(if_scorer, "version", "") != "isolation-forest@2":
            raise RuntimeError("Phase 2C requires isolation-forest@2")

        cal_ae = ae.score_details(model_matrix(family_calibration_legit))
        cal_if = if_scorer.score_details(
            model_matrix(family_calibration_legit), family_calibration_legit["cohort"]
        )
        cal_families = family_calibration_legit["scenario_family"].astype(str).to_numpy()

        policy_probs = bundle.predict_proba(model_matrix(inner_policy), inner_policy["cohort"])
        policy_base = _threshold_actions(
            policy_probs, bundle.thresholds.release, bundle.thresholds.escalate
        )
        policy_ae = ae.score_details(model_matrix(inner_policy))
        policy_if = if_scorer.score_details(model_matrix(inner_policy), inner_policy["cohort"])

        candidate_results: list[dict[str, Any]] = []
        for alpha in ALPHAS:
            recon_cal = family_robust_upper_tail(
                cal_ae.reconstruction_error,
                cal_families,
                target_false_alarm_rate=alpha,
            )
            latent_cal = family_robust_upper_tail(
                cal_ae.latent_mahalanobis_sq,
                cal_families,
                target_false_alarm_rate=alpha,
            )
            if_cal = family_robust_upper_tail(
                cal_if.raw_scores,
                cal_families,
                target_false_alarm_rate=alpha,
            )

            recon_signal = policy_ae.reconstruction_error >= recon_cal.threshold
            latent_signal = policy_ae.latent_mahalanobis_sq >= latent_cal.threshold
            isolation_signal = policy_if.raw_scores >= if_cal.threshold

            for rule in CONSENSUS_RULES:
                signal = consensus_signal(rule, recon_signal, latent_signal, isolation_signal)
                guarded = apply_evidence_only_guard(policy_base, signal)
                metrics = _operational_metrics(inner_policy, policy_base, guarded, signal)
                metrics.update(
                    {
                        "alpha": alpha,
                        "rule": rule,
                        "reconstruction_threshold": recon_cal.threshold,
                        "latent_threshold": latent_cal.threshold,
                        "if_threshold": if_cal.threshold,
                        "pseudo_unseen_legitimate_families": list(pseudo_families),
                    }
                )
                metrics["gate_pass"] = _inner_gate_pass(metrics)
                candidate_results.append(metrics)

        selected = _select_inner_candidate(candidate_results)

        held_probs = bundle.predict_proba(model_matrix(held_out), held_out["cohort"])
        held_base = _threshold_actions(
            held_probs, bundle.thresholds.release, bundle.thresholds.escalate
        )
        if selected is None:
            held_signal = np.zeros(len(held_out), dtype=bool)
            held_guarded = held_base.copy()
            selected_info: dict[str, Any] = {
                "guard_enabled": False,
                "selection": "NO_GUARD",
                "reason": "no inner configuration satisfied family-generalized operational gates",
            }
        else:
            held_ae = ae.score_details(model_matrix(held_out))
            held_if = if_scorer.score_details(model_matrix(held_out), held_out["cohort"])
            held_signal = consensus_signal(
                str(selected["rule"]),
                held_ae.reconstruction_error >= float(selected["reconstruction_threshold"]),
                held_ae.latent_mahalanobis_sq >= float(selected["latent_threshold"]),
                held_if.raw_scores >= float(selected["if_threshold"]),
            )
            held_guarded = apply_evidence_only_guard(held_base, held_signal)
            selected_info = {
                "guard_enabled": True,
                "selection": "FAMILY_GENERALIZED_GUARD",
                **{
                    k: selected[k]
                    for k in (
                        "alpha",
                        "rule",
                        "reconstruction_threshold",
                        "latent_threshold",
                        "if_threshold",
                        "guard_expected_cost_units",
                        "base_expected_cost_units",
                        "legitimate_release_diversion_rate",
                        "worst_legitimate_family_release_diversion_rate",
                        "risky_release_rescue_rate",
                        "risk_release_count",
                    )
                },
            }

        fold_summaries.append(
            {
                "held_out_family": held_out_family,
                "true_risk_state": str(held_out.iloc[0]["true_risk_state"]),
                "pseudo_unseen_legitimate_families": list(pseudo_families),
                "calibration_method": bundle.calibration_method,
                "release_threshold": bundle.thresholds.release,
                "escalate_threshold": bundle.thresholds.escalate,
                "ae_best_epoch": ae.best_epoch,
                "ae_validation_loss": ae.best_validation_loss,
                "inner_candidates": candidate_results,
                "selected_guard": selected_info,
            }
        )

        true_idx = _true_labels(held_out)
        for row_index in range(len(held_out)):
            outer_rows.append(
                {
                    "held_out_family": held_out_family,
                    "true_risk_state": str(held_out.loc[row_index, "true_risk_state"]),
                    "operational_target": str(held_out.loc[row_index, "operational_target"]),
                    "merchant_id": str(held_out.loc[row_index, "merchant_id"]),
                    "base_action": str(held_base[row_index]),
                    "guarded_action": str(held_guarded[row_index]),
                    "novelty_signal": bool(held_signal[row_index]),
                    "guard_enabled_for_fold": bool(selected is not None),
                    "selected_rule": None if selected is None else str(selected["rule"]),
                    "selected_alpha": None if selected is None else float(selected["alpha"]),
                    "true_label_index": int(true_idx[row_index]),
                    "base_action_index": int(LABEL_TO_INDEX[HoldDecision(held_base[row_index])]),
                    "guarded_action_index": int(
                        LABEL_TO_INDEX[HoldDecision(held_guarded[row_index])]
                    ),
                }
            )

    outer_frame = pd.DataFrame(outer_rows)
    outer_frame.to_csv(output / "outer_unknown_family_case_results.csv", index=False)
    overall = _operational_metrics_from_case_frame(outer_frame)
    family_metrics = _family_metrics(outer_frame)
    guard_enabled_folds = sum(bool(f["selected_guard"]["guard_enabled"]) for f in fold_summaries)

    result: dict[str, Any] = {
        "status": "OK",
        "scope": SCOPE,
        "created_at": datetime.now(UTC).isoformat(),
        "promotion_eligible": False,
        "serving_change_authorized": False,
        "production_action_eligible": False,
        "current_serving_champion": "xgb-if-settlement@2",
        "router_version": ROUTER_VERSION,
        "dataset_manifest_sha256": manifest.content_sha256,
        "operational_gates": INNER_GATES,
        "friction_penalty_units": FRICTION_PENALTY,
        "guard_enabled_fold_count": guard_enabled_folds,
        "guard_disabled_fold_count": len(fold_summaries) - guard_enabled_folds,
        "outer_metrics": overall,
        "family_metrics": family_metrics,
        "folds": fold_summaries,
        "interpretation_contract": {
            "novelty": "uncertainty signal, never a fraud label",
            "router": "only RELEASE can be diverted, and only to EVIDENCE_NEEDED",
            "pseudo_unseen_calibration": "novelty thresholds are calibrated with legitimate families excluded from novelty-model fitting",
            "no_guard": "if inner family-generalized gates fail, the fold disables the novelty router rather than forcing an unsafe configuration",
            "promotion": "outer results are synthetic research measurements and cannot authorize promotion",
        },
    }
    result["report_sha256"] = _sha256_json(result)
    _write_json(output / "phase2c_family_generalization_report.json", result)

    print(
        json.dumps(
            {
                "status": result["status"],
                "scope": result["scope"],
                "promotion_eligible": False,
                "guard_enabled_fold_count": result["guard_enabled_fold_count"],
                "guard_disabled_fold_count": result["guard_disabled_fold_count"],
                **overall,
                "report_sha256": result["report_sha256"],
                "output": str(output),
            },
            indent=2,
        )
    )

    print("\nPhase 2C outer unknown-family comparison:\n")
    print(
        pd.DataFrame(
            [
                {
                    "role": "BASE",
                    "expected_cost_units": overall["base_expected_cost_units"],
                    "false_release_rate": overall["base_false_release_rate"],
                    "unknown_legitimate_release_diversion_rate": 0.0,
                    "unknown_risk_release_rescue_rate": 0.0,
                },
                {
                    "role": "FAMILY_GENERALIZED_GUARD",
                    "expected_cost_units": overall["guard_expected_cost_units"],
                    "false_release_rate": overall["guard_false_release_rate"],
                    "unknown_legitimate_release_diversion_rate": overall[
                        "legitimate_release_diversion_rate"
                    ],
                    "unknown_risk_release_rescue_rate": overall["risky_release_rescue_rate"],
                },
            ]
        ).to_string(index=False)
    )

    print("\nPer-family outer results:\n")
    print(pd.DataFrame(family_metrics).to_string(index=False))

    print("\nPer-fold selected research guards:\n")
    selection_rows = []
    for fold in fold_summaries:
        selected = fold["selected_guard"]
        selection_rows.append(
            {
                "held_out_family": fold["held_out_family"],
                "true_risk_state": fold["true_risk_state"],
                "guard_enabled": selected["guard_enabled"],
                "rule": selected.get("rule"),
                "alpha": selected.get("alpha"),
                "pseudo_legit": ",".join(fold["pseudo_unseen_legitimate_families"]),
            }
        )
    print(pd.DataFrame(selection_rows).to_string(index=False))


def _choose_pseudo_unseen_legitimate_families(
    *, train: pd.DataFrame, calibration: pd.DataFrame, held_out_family: str, rotation: int
) -> tuple[str, ...]:
    available = []
    for family in LEGITIMATE_FAMILIES:
        if family == held_out_family:
            continue
        train_n = int(np.sum(train["scenario_family"].astype(str).to_numpy() == family))
        cal_n = int(np.sum(calibration["scenario_family"].astype(str).to_numpy() == family))
        if train_n >= 5 and cal_n >= 5:
            available.append(family)
    if len(available) < 2:
        raise RuntimeError("Phase 2C requires at least two pseudo-unseen legitimate families")
    start = (rotation * 2) % len(available)
    first = available[start]
    second = available[(start + 1) % len(available)]
    if second == first:
        second = available[(start + 2) % len(available)]
    return (first, second)


def _select_inner_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    safe = [row for row in rows if bool(row["gate_pass"])]
    if not safe:
        return None
    return min(
        safe,
        key=lambda row: (
            float(row["adjusted_cost_with_friction"]),
            float(row["guard_expected_cost_units"]),
            float(row["legitimate_release_diversion_rate"]),
            -float(row["risky_release_rescue_rate"]),
        ),
    )


def _inner_gate_pass(metrics: dict[str, Any]) -> bool:
    rescue_ok = (
        int(metrics["risk_release_count"]) == 0
        or float(metrics["risky_release_rescue_rate"])
        >= INNER_GATES["min_risky_release_rescue_rate_when_applicable"]
    )
    return bool(
        float(metrics["legitimate_release_diversion_rate"])
        <= INNER_GATES["max_legitimate_release_diversion_rate"]
        and float(metrics["worst_legitimate_family_release_diversion_rate"])
        <= INNER_GATES["max_worst_legitimate_family_release_diversion_rate"]
        and rescue_ok
        and float(metrics["guard_expected_cost_units"])
        <= float(metrics["base_expected_cost_units"])
    )


def _operational_metrics(
    frame: pd.DataFrame,
    base_actions: np.ndarray,
    guarded_actions: np.ndarray,
    signal: np.ndarray,
) -> dict[str, Any]:
    true_idx = _true_labels(frame)
    base_idx = np.asarray([LABEL_TO_INDEX[HoldDecision(v)] for v in base_actions], dtype=int)
    guard_idx = np.asarray([LABEL_TO_INDEX[HoldDecision(v)] for v in guarded_actions], dtype=int)
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix, dtype=float)
    legit = frame["true_risk_state"].astype(str).to_numpy() == TrueRiskState.LEGITIMATE
    risk = ~legit
    legit_release = legit & (base_actions == HoldDecision.RELEASE)
    risk_release = risk & (base_actions == HoldDecision.RELEASE)
    non_release_truth = true_idx != LABEL_TO_INDEX[HoldDecision.RELEASE]

    family_diversions: list[float] = []
    for family in sorted(frame["scenario_family"].astype(str).unique()):
        mask = frame["scenario_family"].astype(str).to_numpy() == family
        if not np.all(legit[mask]):
            continue
        release_mask = mask & (base_actions == HoldDecision.RELEASE)
        if np.any(release_mask):
            family_diversions.append(float(np.mean(signal[release_mask])))

    diverted_legit = int(np.count_nonzero(signal & legit_release))
    adjusted = float(
        np.mean(matrix[guard_idx, true_idx])
    ) + FRICTION_PENALTY * diverted_legit / len(frame)
    return {
        "base_expected_cost_units": round(float(np.mean(matrix[base_idx, true_idx])), 8),
        "guard_expected_cost_units": round(float(np.mean(matrix[guard_idx, true_idx])), 8),
        "adjusted_cost_with_friction": round(adjusted, 8),
        "base_false_release_rate": round(
            float(np.mean(base_idx[non_release_truth] == LABEL_TO_INDEX[HoldDecision.RELEASE]))
            if np.any(non_release_truth)
            else 0.0,
            8,
        ),
        "guard_false_release_rate": round(
            float(np.mean(guard_idx[non_release_truth] == LABEL_TO_INDEX[HoldDecision.RELEASE]))
            if np.any(non_release_truth)
            else 0.0,
            8,
        ),
        "legitimate_release_diversion_rate": round(
            float(np.mean(signal[legit_release])) if np.any(legit_release) else 0.0, 8
        ),
        "worst_legitimate_family_release_diversion_rate": round(
            max(family_diversions) if family_diversions else 0.0, 8
        ),
        "risky_release_rescue_rate": round(
            float(np.mean(signal[risk_release])) if np.any(risk_release) else 0.0, 8
        ),
        "risk_release_count": int(np.count_nonzero(risk_release)),
        "legitimate_release_count": int(np.count_nonzero(legit_release)),
        "novelty_signal_rate": round(float(np.mean(signal)), 8),
    }


def _operational_metrics_from_case_frame(frame: pd.DataFrame) -> dict[str, Any]:
    base_actions = frame["base_action"].astype(str).to_numpy()
    signal = frame["novelty_signal"].astype(bool).to_numpy()
    true_idx = frame["true_label_index"].to_numpy(dtype=int)
    base_idx = frame["base_action_index"].to_numpy(dtype=int)
    guard_idx = frame["guarded_action_index"].to_numpy(dtype=int)
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix, dtype=float)
    legit = frame["true_risk_state"].astype(str).to_numpy() == TrueRiskState.LEGITIMATE
    risk = ~legit
    legit_release = legit & (base_actions == str(HoldDecision.RELEASE))
    risk_release = risk & (base_actions == str(HoldDecision.RELEASE))
    non_release_truth = true_idx != LABEL_TO_INDEX[HoldDecision.RELEASE]
    return {
        "base_expected_cost_units": round(float(np.mean(matrix[base_idx, true_idx])), 8),
        "guard_expected_cost_units": round(float(np.mean(matrix[guard_idx, true_idx])), 8),
        "base_false_release_rate": round(
            float(np.mean(base_idx[non_release_truth] == LABEL_TO_INDEX[HoldDecision.RELEASE]))
            if np.any(non_release_truth)
            else 0.0,
            8,
        ),
        "guard_false_release_rate": round(
            float(np.mean(guard_idx[non_release_truth] == LABEL_TO_INDEX[HoldDecision.RELEASE]))
            if np.any(non_release_truth)
            else 0.0,
            8,
        ),
        "unknown_legitimate_signal_rate": round(float(np.mean(signal[legit])), 8),
        "unknown_risk_signal_rate": round(float(np.mean(signal[risk])), 8),
        "legitimate_release_diversion_rate": round(
            float(np.mean(signal[legit_release])) if np.any(legit_release) else 0.0, 8
        ),
        "risky_release_rescue_rate": round(
            float(np.mean(signal[risk_release])) if np.any(risk_release) else 0.0, 8
        ),
        "unknown_legitimate_release_count": int(np.count_nonzero(legit_release)),
        "unknown_risk_release_count": int(np.count_nonzero(risk_release)),
    }


def _family_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix, dtype=float)
    for family in sorted(frame["held_out_family"].astype(str).unique()):
        group = frame[frame["held_out_family"].astype(str) == family]
        base_idx = group["base_action_index"].to_numpy(dtype=int)
        guard_idx = group["guarded_action_index"].to_numpy(dtype=int)
        true_idx = group["true_label_index"].to_numpy(dtype=int)
        base_release = group["base_action"].astype(str).to_numpy() == str(HoldDecision.RELEASE)
        signal = group["novelty_signal"].astype(bool).to_numpy()
        output.append(
            {
                "held_out_family": family,
                "true_risk_state": str(group.iloc[0]["true_risk_state"]),
                "n": len(group),
                "guard_enabled": bool(group.iloc[0]["guard_enabled_for_fold"]),
                "selected_rule": group.iloc[0]["selected_rule"],
                "selected_alpha": group.iloc[0]["selected_alpha"],
                "novelty_signal_rate": round(float(np.mean(signal)), 8),
                "base_release_rate": round(float(np.mean(base_release)), 8),
                "release_diversion_or_rescue_rate": round(
                    float(np.mean(signal[base_release])) if np.any(base_release) else 0.0, 8
                ),
                "base_expected_cost_units": round(float(np.mean(matrix[base_idx, true_idx])), 8),
                "guard_expected_cost_units": round(float(np.mean(matrix[guard_idx, true_idx])), 8),
            }
        )
    return output


def _threshold_actions(
    probabilities: np.ndarray, release_threshold: float, escalate_threshold: float
) -> np.ndarray:
    actions = np.full(len(probabilities), HoldDecision.EVIDENCE_NEEDED, dtype=object)
    release = probabilities[:, 0] >= release_threshold
    escalate = (~release) & (probabilities[:, 2] >= escalate_threshold)
    actions[release] = HoldDecision.RELEASE
    actions[escalate] = HoldDecision.ESCALATE
    return actions


def _true_labels(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [LABEL_TO_INDEX[HoldDecision(value)] for value in frame["operational_target"]],
        dtype=int,
    )


def _partition(frame: pd.DataFrame, merchants: list[str]) -> pd.DataFrame:
    result = frame[frame["merchant_id"].isin(merchants)].reset_index(drop=True)
    if result.empty:
        raise RuntimeError("empty merchant partition")
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
