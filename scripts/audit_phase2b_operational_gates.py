from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FRICTION_PENALTIES = (0.0, 5.0, 10.0, 20.0)
DEFAULT_GATES = {
    "max_guard_false_release_rate": 0.05,
    "max_unknown_legitimate_release_diversion_rate": 0.20,
    "max_worst_legitimate_family_release_diversion_rate": 0.50,
    "min_unknown_risk_release_rescue_rate": 0.50,
    "require_non_increasing_expected_cost": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase 2B novelty candidates against explicit operational gates"
    )
    parser.add_argument("--project-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    report_path = args.report.resolve() if args.report else _latest_report(root)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = [dict(row) for row in report.get("candidate_metrics", [])]
    if not candidates:
        raise RuntimeError(f"no candidate_metrics in {report_path}")

    total_cases = sum(int(fold.get("n", 0)) for fold in report.get("folds", []))
    if total_cases <= 0:
        raise RuntimeError("could not determine total Phase 2B case count")

    evaluated = []
    for row in candidates:
        base_cost = float(row["base_expected_cost_units"])
        guard_cost = float(row["guard_expected_cost_units"])
        false_release = float(row["guard_false_release_rate"])
        legit_diversion = float(row["unknown_legitimate_release_diversion_rate"])
        worst_legit_diversion = float(
            row.get("worst_legitimate_family_release_diversion_rate", legit_diversion)
        )
        risk_rescue = float(row["unknown_risk_release_rescue_rate"])
        min_risk_family_rescue = float(row.get("minimum_risk_family_release_rescue_rate", 0.0))
        legit_release_count = int(row.get("unknown_legitimate_release_count", 0))
        diverted_legit = legit_release_count * legit_diversion

        gate_checks = {
            "false_release": false_release <= DEFAULT_GATES["max_guard_false_release_rate"],
            "legitimate_diversion": legit_diversion
            <= DEFAULT_GATES["max_unknown_legitimate_release_diversion_rate"],
            "worst_legitimate_family": worst_legit_diversion
            <= DEFAULT_GATES["max_worst_legitimate_family_release_diversion_rate"],
            "risk_rescue": risk_rescue >= DEFAULT_GATES["min_unknown_risk_release_rescue_rate"],
            "expected_cost": guard_cost <= base_cost,
        }
        gate_pass = all(gate_checks.values())
        sensitivity = {}
        for penalty in FRICTION_PENALTIES:
            adjusted = guard_cost + penalty * diverted_legit / total_cases
            sensitivity[f"evidence_friction_{penalty:g}"] = round(adjusted, 8)

        evaluated.append(
            {
                **row,
                "gate_pass": gate_pass,
                "gate_checks": gate_checks,
                "estimated_legitimate_release_diversions": round(diverted_legit, 4),
                "operational_cost_sensitivity": sensitivity,
                "minimum_risk_family_release_rescue_rate": min_risk_family_rescue,
            }
        )

    safe = [row for row in evaluated if row["gate_pass"]]
    safe_sorted = sorted(
        safe,
        key=lambda r: (
            float(r["operational_cost_sensitivity"]["evidence_friction_10"]),
            float(r["guard_false_release_rate"]),
            float(r["unknown_legitimate_release_diversion_rate"]),
            -float(r["unknown_risk_release_rescue_rate"]),
        ),
    )

    threshold_groups: dict[str, list[tuple[float, tuple[float, ...]]]] = {}
    for row in evaluated:
        rule = str(row["consensus_rule"])
        signature = (
            float(row["guard_expected_cost_units"]),
            float(row["guard_false_release_rate"]),
            float(row["unknown_legitimate_release_diversion_rate"]),
            float(row["unknown_risk_release_rescue_rate"]),
        )
        threshold_groups.setdefault(rule, []).append(
            (float(row["target_false_alarm_rate"]), signature)
        )
    threshold_insensitive_rules = []
    for rule, items in threshold_groups.items():
        if len({sig for _, sig in items}) == 1 and len(items) > 1:
            threshold_insensitive_rules.append(rule)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "status": "SAFE_CONFIGURATION_FOUND" if safe_sorted else "NO_SAFE_CONFIGURATION",
        "scope": "SYNTHETIC_RESEARCH_ONLY_PHASE2B1_OPERATIONAL_GATE_AUDIT_NO_PROMOTION",
        "created_at": datetime.now(UTC).isoformat(),
        "source_report": str(report_path),
        "source_report_sha256": _sha256_file(report_path),
        "promotion_eligible": False,
        "serving_change_authorized": False,
        "production_action_eligible": False,
        "current_serving_champion": "xgb-if-settlement@2",
        "operational_gates": DEFAULT_GATES,
        "evidence_friction_penalties": list(FRICTION_PENALTIES),
        "total_cases": total_cases,
        "candidate_count": len(evaluated),
        "safe_candidate_count": len(safe_sorted),
        "threshold_insensitive_rules": sorted(threshold_insensitive_rules),
        "recommended_safe_candidate": safe_sorted[0] if safe_sorted else None,
        "candidates": sorted(
            evaluated,
            key=lambda r: (str(r["consensus_rule"]), float(r["target_false_alarm_rate"])),
        ),
        "interpretation": {
            "no_safe_configuration": "Do not promote a novelty router. The research stack needs family-generalization calibration or a different consensus/representation before runtime integration.",
            "threshold_insensitive": "If a rule has identical outcomes across alpha values, in-distribution false-alarm calibration is not controlling unseen-family behavior.",
            "friction_sensitivity": "Adds an explicit penalty for diverting a legitimate RELEASE to EVIDENCE_NEEDED; it is a research sensitivity analysis, not a production cost estimate.",
        },
    }
    result["report_sha256"] = _sha256_json(result)
    report_out = output / "phase2b1_operational_gate_audit.json"
    report_out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "safe_candidate_count": result["safe_candidate_count"],
                "threshold_insensitive_rules": result["threshold_insensitive_rules"],
                "recommended_rule": None
                if result["recommended_safe_candidate"] is None
                else result["recommended_safe_candidate"]["consensus_rule"],
                "recommended_alpha": None
                if result["recommended_safe_candidate"] is None
                else result["recommended_safe_candidate"]["target_false_alarm_rate"],
                "report_sha256": result["report_sha256"],
                "output": str(output),
            },
            indent=2,
        )
    )

    print("\nAll Phase 2B candidates under explicit operational gates:\n")
    columns = [
        "target_false_alarm_rate",
        "consensus_rule",
        "guard_expected_cost_units",
        "guard_false_release_rate",
        "unknown_legitimate_release_diversion_rate",
        "worst_legitimate_family_release_diversion_rate",
        "unknown_risk_release_rescue_rate",
        "minimum_risk_family_release_rescue_rate",
        "gate_pass",
    ]
    try:
        import pandas as pd

        print(
            pd.DataFrame(evaluated)[columns]
            .sort_values(["consensus_rule", "target_false_alarm_rate"])
            .to_string(index=False)
        )
    except Exception:
        for row in evaluated:
            print({key: row.get(key) for key in columns})

    print("\nEvidence-friction sensitivity (10 cost-units per diverted legitimate RELEASE):\n")
    try:
        import pandas as pd

        sens = pd.DataFrame(
            [
                {
                    "alpha": r["target_false_alarm_rate"],
                    "rule": r["consensus_rule"],
                    "base_cost": r["base_expected_cost_units"],
                    "guard_cost": r["guard_expected_cost_units"],
                    "adjusted_cost_friction10": r["operational_cost_sensitivity"][
                        "evidence_friction_10"
                    ],
                    "legit_diversion": r["unknown_legitimate_release_diversion_rate"],
                    "risk_rescue": r["unknown_risk_release_rescue_rate"],
                }
                for r in evaluated
            ]
        )
        print(
            sens.sort_values(["adjusted_cost_friction10", "legit_diversion"]).to_string(index=False)
        )
    except Exception:
        pass


def _latest_report(root: Path) -> Path:
    base = root / "artifacts" / "research"
    matches = sorted(
        base.glob("phase2b-unknown-family-*/phase2b_unknown_family_report.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"no Phase 2B report under {base}")
    return matches[0]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
