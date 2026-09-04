from __future__ import annotations

from pathlib import Path

from .research_status import FINAL_V3_RESEARCH_STATUS

CHAMPION_MODEL_VERSION = "xgb-if-settlement@2"
CHAMPION_RELEASE_PATH = "artifacts/tier0/model-release"
CHAMPION_PUBLIC_KEY_PATH = "artifacts/tier0/model-release-public-key.txt"


def model_governance_status(
    project_root: Path | None = None, *, decision_mode: str
) -> dict[str, object]:
    root = project_root or Path.cwd()
    release = root / CHAMPION_RELEASE_PATH
    public_key = root / CHAMPION_PUBLIC_KEY_PATH
    return {
        "registry_champion": {
            "model_version": CHAMPION_MODEL_VERSION,
            "release_path": CHAMPION_RELEASE_PATH,
            "public_key_path": CHAMPION_PUBLIC_KEY_PATH,
            "release_present": release.exists(),
            "public_key_present": public_key.exists(),
            "production_action_eligible": False,
            "promotion_state": "FROZEN_CHAMPION_RESEARCH_ONLY",
        },
        "active_enforcement_runtime": {
            "decision_mode": decision_mode,
            "runtime": (
                "human-only@1" if decision_mode == "human_only" else "configured-model-runtime"
            ),
            "automatic_release_enabled": False if decision_mode == "human_only" else None,
        },
        "challengers": [
            {
                "name": "graphsage-risk@1",
                "phase": 3,
                "status": "IMPLEMENTED_RESEARCH_CHALLENGER",
                "auto_promotion": False,
            },
            {
                "name": "sequence-lstm-risk@1",
                "phase": 4,
                "status": "IMPLEMENTED_RESEARCH_CHALLENGER",
                "auto_promotion": False,
            },
        ],
        "promotion_gate": {
            "automatic_promotion": False,
            "requires_fixed_safety_gate": True,
            "requires_human_approval": True,
            "latest_research_decision": FINAL_V3_RESEARCH_STATUS["decision"],
        },
    }
