import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from unittest.mock import MagicMock
from causal_validation.geo_matcher import GeoMatcher

# Mock Anthropic client
mock_client = MagicMock()
mock_response = MagicMock()
mock_response.content = [MagicMock()]
mock_response.content[0].text = json.dumps(
    {
        "ranked_geo_ids": ["geo_A", "geo_B", "geo_D", "geo_C", "geo_X"],
        "reasoning": "geo_A and geo_B are most similar based on urbanization and income.",
    }
)
mock_client.messages.create.return_value = mock_response

matcher = GeoMatcher(anthropic_client=mock_client)

treated_desc = "Urban, high income, Northeast US, population 2M"
candidates = [
    {"geo_id": "geo_A", "description": "Urban, high income, Mid-Atlantic"},
    {"geo_id": "geo_B", "description": "Suburban, medium income, Northeast"},
    {"geo_id": "geo_C", "description": "Rural, low income, Southeast"},
    {"geo_id": "geo_D", "description": "Urban, medium income, Midwest"},
]

result = matcher.match(treated_desc, candidates)

print(f"\nRanked geos: {result}")

# Checks
assert isinstance(result, list), "Result should be a list"
assert len(result) <= 5, "Should return at most 5 geos"
assert (
    "geo_X" not in result
), "Hallucinated geo_X should be filtered out (not in candidate list)"
assert all(
    g in {c["geo_id"] for c in candidates} for g in result
), "All returned IDs must be from candidate list"
print("geo_matcher — hallucinated IDs filtered, valid IDs returned")

# Test caching
result2 = matcher.match(treated_desc, candidates)
assert (
    mock_client.messages.create.call_count == 1
), "Second call with same description should use cache, not call API again"
print("geo_matcher — caching works, API called only once")

# Test empty candidates
empty_result = matcher.match(treated_desc, [])
assert empty_result == [], "Empty candidates should return empty list"
print("geo_matcher — empty candidates handled")
