from __future__ import annotations

FINAL_V3_RESEARCH_STATUS = {
    "version": "v3C-final-cycle-20260903",
    "decision": "FINAL_ML_STOP_NO_SAFE_RELEASE_CANDIDATE",
    "stopping_decision": "STOP_ML",
    "gate_passed": False,
    "false_release_rate": 0.07333333,
    "true_release_recall": 0.43333333,
    "maximum_false_release_rate": 0.05,
    "minimum_true_release_recall": 0.20,
    "sealed_accessed": False,
    "production_data_used": False,
    "production_action_eligible": False,
    "human_approval_required": True,
}


PHASE_IMPLEMENTATION_STATUS = {
    "phase_1_core": "COMPLETE",
    "phase_2_novelty": "COMPLETE_RESEARCH_NOT_PROMOTED",
    "phase_3_graph_gnn": "IMPLEMENTED_RESEARCH_CHALLENGER",
    "phase_4_sequence": "IMPLEMENTED_RESEARCH_CHALLENGER",
    "phase_5_drift": "IMPLEMENTED_MONITORING_FOUNDATION",
    "phase_6_uncertainty": "IMPLEMENTED_RESEARCH_FOUNDATION",
    "phase_7_product": "COMPLETE_HUMAN_GATED",
}
