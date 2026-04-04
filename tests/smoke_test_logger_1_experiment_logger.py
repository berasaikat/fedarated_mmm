# tests/smoke_test_phase9_1_experiment_logger.py
import json
import tempfile
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.experiment_logger import ExperimentLogger

# Setup temp output dir
tmp_dir = tempfile.mkdtemp()

try:
    logger = ExperimentLogger(experiment_id="test_exp_001", output_dir=tmp_dir)

    # Test 1 — directory structure created
    assert (Path(tmp_dir) / "test_exp_001" / "logs").exists()
    print("Test 1 — log directory created")

    # Test 2 — log_round writes to rounds.jsonl
    global_summary = {
        "paid_search": {"mean": 0.35, "std": 0.08},
        "social": {"mean": 0.18, "std": 0.10},
    }
    surprise_scores = {"participant_1": {"paid_search": 0.3, "social": 0.1}}

    logger.log_round(
        round_num=1,
        global_summary=global_summary,
        surprise_scores=surprise_scores,
        epsilon_spent_per_participant=0.5,
        num_active_participants=3,
    )

    with open(logger.rounds_log_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    assert len(lines) == 1
    assert lines[0]["round_num"] == 1
    assert lines[0]["epsilon_spent_total"] == 1.5  # 0.5 * 3
    assert lines[0]["num_active_participants"] == 3
    assert "global_summary" in lines[0]
    assert "surprise_scores" in lines[0]
    print("Test 2 — log_round writes correct JSONL with total epsilon")

    # Test 3 — log_priors writes to priors.jsonl
    priors = {
        "paid_search": {"mu": 0.3, "sigma": 0.1},
        "social": {"mu": 0.2, "sigma": 0.1},
    }
    logger.log_priors(round_num=1, participant_id="participant_1", priors_dict=priors)

    with open(logger.priors_log_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    assert len(lines) == 1
    assert lines[0]["participant_id"] == "participant_1"
    assert lines[0]["round_num"] == 1
    assert "priors" in lines[0]
    print("Test 3 — log_priors writes correct JSONL")

    # Test 4 — log_audit writes to audits.jsonl
    audit_result = {
        "channel": "paid_search",
        "mmm_beta_mean": 0.35,
        "mmm_beta_ci": [0.25, 0.45],
        "att_estimate_normalized": 0.33,
        "coverage": True,
        "gap": -0.02,
    }
    logger.log_audit(audit_result)

    with open(logger.audits_log_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    assert len(lines) == 1
    assert lines[0]["audit_result"]["channel"] == "paid_search"
    assert lines[0]["audit_result"]["coverage"] == True
    print("Test 4 — log_audit writes correct JSONL")

    # Test 5 — summary metrics accumulate correctly
    logger.log_round(
        round_num=2,
        global_summary=global_summary,
        surprise_scores=surprise_scores,
        epsilon_spent_per_participant=0.5,
        num_active_participants=3,
    )
    logger.log_priors(round_num=2, participant_id="participant_2", priors_dict=priors)

    assert logger.summary["metrics"]["total_rounds_logged"] == 2
    assert logger.summary["metrics"]["total_priors_elicited"] == 2
    assert logger.summary["metrics"]["total_audits_logged"] == 1
    assert abs(logger.summary["metrics"]["epsilon_spent_cumulated"] - 3.0) < 1e-9
    print("Test 5 — summary metrics accumulate correctly (2 rounds × 1.5 = 3.0 eps)")

    # Test 6 — save_summary writes experiment_summary.json
    logger.save_summary()
    summary_path = Path(tmp_dir) / "test_exp_001" / "experiment_summary.json"
    assert summary_path.exists()

    with open(summary_path) as f:
        saved = json.load(f)

    assert saved["experiment_id"] == "test_exp_001"
    assert "datetime_start" in saved
    assert "datetime_end" in saved
    assert saved["metrics"]["total_rounds_logged"] == 2
    assert abs(saved["metrics"]["epsilon_spent_cumulated"] - 3.0) < 1e-9
    print("Test 6 — save_summary writes valid experiment_summary.json")

    # Test 7 — JSONL files are append-only (multiple rounds accumulate)
    with open(logger.rounds_log_path) as f:
        all_rounds = [json.loads(l) for l in f if l.strip()]
    assert len(all_rounds) == 2
    assert all_rounds[0]["round_num"] == 1
    assert all_rounds[1]["round_num"] == 2
    print("Test 7 — JSONL files append correctly across multiple rounds")

    # Test 8 — read-back methods work
    rounds = logger.read_rounds()
    priors_ = logger.read_priors()
    audits = logger.read_audits()

    assert len(rounds) == 2
    assert len(priors_) == 2
    assert len(audits) == 1
    print("Test 8 — read-back methods return correct entry counts")

    # Test 9 — numpy floats serialize without error
    import numpy as np

    np_summary = {"paid_search": {"mean": np.float32(0.35), "std": np.float64(0.08)}}
    logger.log_round(
        round_num=3,
        global_summary=np_summary,
        surprise_scores={},
        epsilon_spent_per_participant=0.5,
        num_active_participants=1,
    )
    with open(logger.rounds_log_path) as f:
        last = [json.loads(l) for l in f if l.strip()][-1]
    assert abs(last["global_summary"]["paid_search"]["mean"] - 0.35) < 1e-4
    print("Test 9 — numpy floats serialize correctly via default=float")

finally:
    shutil.rmtree(tmp_dir)

print("\nAll ExperimentLogger tests passed")
