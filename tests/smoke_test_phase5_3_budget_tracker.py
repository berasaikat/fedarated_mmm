import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from privacy.budget_tracker import PrivacyBudgetTracker, PrivacyBudgetExhausted

# Test 1 — basic spend and remaining
tracker = PrivacyBudgetTracker(
    total_epsilon=1.0,
    total_delta=1e-5,
    participant_ids=["participant_1", "participant_2"]
)

tracker.spend("participant_1", 0.3, 1e-6)
rem_eps, rem_delta = tracker.remaining("participant_1")
assert abs(rem_eps - 0.7) < 1e-10, f"Expected 0.7 remaining, got {rem_eps}"
print("Spend and remaining work correctly")

# Test 2 — independent budgets per participant
rem_eps_p2, _ = tracker.remaining("participant_2")
assert abs(rem_eps_p2 - 1.0) < 1e-10, "P2 budget should be unaffected by P1 spend"
print("Participant budgets are independent")

# Test 3 — budget exhaustion raises exception
tracker2 = PrivacyBudgetTracker(
    total_epsilon=0.5,
    total_delta=1e-5,
    participant_ids=["participant_1"]
)
tracker2.spend("participant_1", 0.4, 1e-6)

try:
    tracker2.spend("participant_1", 0.2, 1e-6)  # would exceed 0.5
    assert False, "Should have raised PrivacyBudgetExhausted"
except PrivacyBudgetExhausted as e:
    print(f"Budget exhaustion raised correctly: {str(e)[:60]}...")

# Test 4 — is_exhausted flag
tracker3 = PrivacyBudgetTracker(
    total_epsilon=0.3,
    total_delta=1e-5,
    participant_ids=["participant_1"]
)
assert not tracker3.is_exhausted("participant_1")
tracker3.spend("participant_1", 0.3, 1e-6)
assert tracker3.is_exhausted("participant_1")
print("is_exhausted works correctly")

# Test 5 — multi-round accumulation (5 rounds of 0.2 each on budget of 1.0)
tracker4 = PrivacyBudgetTracker(
    total_epsilon=1.0,
    total_delta=1e-5,
    participant_ids=["participant_1"]
)
for round_num in range(5):
    tracker4.spend("participant_1", 0.2, 1e-6)
    rem, _ = tracker4.remaining("participant_1")
    print(f"  Round {round_num+1}: remaining epsilon = {rem:.2f}")

assert tracker4.is_exhausted("participant_1")
print("Budget correctly exhausted after 5 rounds")