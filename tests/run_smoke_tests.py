import subprocess
import sys
from pathlib import Path

tests_dir = Path(__file__).parent

# Define smoke tests in order
smoke_tests = [
    ("MMM Model", "smoke_test_mmm_model.py"),
    ("Prompt Builder", "smoke_test_prompt_construction_1_prompt_builder.py"),
    ("Validator", "smoke_test_prompt_construction_2_validator.py"),
    ("Surprise", "smoke_test_prompt_construction_3_surprise.py"),
    ("API Call", "smoke_test_prompt_construction_4_api_call.py"),
    ("Feedback Loop", "smoke_test_prompt_construction_5_feedback_loop.py"),
    ("DP Noise - Dependency", "smoke_test_dp_noise_1_dependency.py"),
    ("DP Noise - Sensitivity", "smoke_test_dp_noise_2_sensitivity.py"),
    ("DP Noise - Budget Tracker", "smoke_test_dp_noise_3_budget_tracker.py"),
    ("DP Noise - Posterior", "smoke_test_dp_noise_4_posterior.py"),
    (
        "DP Noise - Statistical Validity",
        "smoke_test_dp_noise_5_statistical_validity.py",
    ),
    ("Aggregator - FedAvg", "smoke_test_aggregator_1_fed_avg.py"),
    ("Aggregator - Hierarchical", "smoke_test_aggregator_2_hierarchical.py"),
    ("Aggregator - Round Manager", "smoke_test_aggregator_3_round_manager.py"),
    ("Aggregator - Federated Loop 1", "smoke_test_aggregator_4_fedarated_loop_1.py"),
    ("Aggregator - Federated Loop 2", "smoke_test_aggregator_4_fedarated_loop_2.py"),
    ("Flower - MMM Client", "smoke_test_flower_1_MMMClient .py"),
    ("Flower - LLM Prior", "smoke_test_flower_2_llm_prior.py"),
    ("Flower - Budget", "smoke_test_flower_3_budget.py"),
    ("Flower - Strategy", "smoke_test_flower_4_strategy.py"),
    ("Flower - Full Simulation", "smoke_test_flower_5_full_simulation.py"),
    ("Control - Geo Loader", "smoke_test_control_1_geo_loader.py"),
    ("Control - Fit Synthetic", "smoke_test_control_2_fit_synthetic.py"),
    ("Control - Estimate Increment", "smoke_test_control_3_estimate_increment.py"),
    ("Control - Geo Matcher", "smoke_test_control_4_geo_matcher.py"),
    ("Control - Full Audit", "smoke_test_control_5_full_audit.py"),
    ("Logger - Experiment Logger", "smoke_test_logger_1_experiment_logger.py"),
    ("Logger - Convergence", "smoke_test_logger_2_convergence.py"),
    ("Visualization", "smoke_test_visualization.py"),
]

results = []

for label, filename in smoke_tests:
    filepath = tests_dir / filename
    if not filepath.exists():
        results.append((label, "SKIPPED", "file not found"))
        continue

    print(f"\n{'='*50}")
    print(f"Running: {label}")
    print(f"{'='*50}")

    result = subprocess.run(
        [sys.executable, str(filepath)], capture_output=False  # streams output live
    )

    status = "PASSED" if result.returncode == 0 else "FAILED"
    results.append((label, status, ""))

# Print summary
print(f"\n{'='*50}")
print("SMOKE TEST SUMMARY")
print(f"{'='*50}")
for label, status, note in results:
    icon = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "–"
    note_str = f"  ({note})" if note else ""
    print(f"  {icon} {status:<8} {label}{note_str}")

# Exit with error if any failed
failed = [r for r in results if r[1] == "FAILED"]
if failed:
    print(f"\n{len(failed)} test(s) failed.")
    sys.exit(1)
else:
    print(f"\nAll tests passed.")
