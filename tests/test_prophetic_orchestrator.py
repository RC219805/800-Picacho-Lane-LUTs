“”“Comprehensive tests for the whimsical future-proofing helpers.

This test suite covers:

- Input normalization and parsing
- Temporal antibody deployment
- Probability tracking
- Edge cases and error handling
- Integration scenarios
  “””

from **future** import annotations

import pytest

from prophetic_orchestrator import (
CausalityEngine,
ProbabilityWeaver,
PropheticOrchestrator,
TemporalAntibody,
WeakPoint,
)

# ============================================================================

# Original Tests (from the provided test file)

# ============================================================================

def test_causality_engine_normalizes_inputs() -> None:
“”“Original test: verify input normalization.”””
engine = CausalityEngine()
predicted_failure = {
“weak_points”: [
{
“component”: “database”,
“failure_mode”: “replication lag”,
“severity”: “high”,
},
“api:timeout”,
WeakPoint(component=“cache”, failure_mode=“eviction storm”, severity=“low”),
]
}

```
result = engine.trace_failure_origins(predicted_failure)

assert [wp.component for wp in result] == ["database", "api", "cache"]
assert [wp.failure_mode for wp in result] == [
    "replication lag",
    "timeout",
    "eviction storm",
]
assert [wp.severity for wp in result] == ["high", None, "low"]
```

def test_prophetic_orchestrator_deploys_temporal_antibodies() -> None:
“”“Original test: verify antibody deployment and tracking.”””
orchestrator = PropheticOrchestrator()
predicted_failure = {
“weak_points”: [
{
“component”: “ingest”,
“failure_mode”: “queue saturation”,
“severity”: “critical”,
},
{
“component”: “renderer”,
“failure_mode”: “color drift”,
“severity”: “medium”,
},
]
}

```
antibodies = orchestrator.prevent_future_failure(predicted_failure)

assert all(isinstance(item, TemporalAntibody) for item in antibodies)
assert orchestrator.deployed_antibodies == antibodies

# The orchestrator itself should have updated the probability field to 0.9999
for deployed in orchestrator.deployed_antibodies:
    probability = orchestrator.probability_weaver.probability_of(deployed.target)
    assert probability == 0.9999

# Sanity-check the generated countermeasures reference the component names.
for antibody in antibodies:
    assert antibody.target.component in antibody.countermeasure
```

# ============================================================================

# Enhanced WeakPoint Tests

# ============================================================================

class TestWeakPoint:
“”“Comprehensive tests for WeakPoint parsing and validation.”””

```
def test_direct_construction(self) -> None:
    wp = WeakPoint(component="database", failure_mode="connection timeout")
    assert wp.component == "database"
    assert wp.failure_mode == "connection timeout"
    assert wp.severity is None

def test_construction_with_severity(self) -> None:
    wp = WeakPoint(
        component="cache", failure_mode="memory leak", severity="critical"
    )
    assert wp.component == "cache"
    assert wp.failure_mode == "memory leak"
    assert wp.severity == "critical"

def test_from_value_passthrough(self) -> None:
    original = WeakPoint(component="api", failure_mode="rate limit")
    result = WeakPoint.from_value(original)
    assert result is original

def test_from_value_dict_complete(self) -> None:
    wp = WeakPoint.from_value(
        {
            "component": "storage",
            "failure_mode": "disk full",
            "severity": "high",
        }
    )
    assert wp.component == "storage"
    assert wp.failure_mode == "disk full"
    assert wp.severity == "high"

def test_from_value_dict_minimal(self) -> None:
    wp = WeakPoint.from_value(
        {"component": "network", "failure_mode": "packet loss"}
    )
    assert wp.component == "network"
    assert wp.failure_mode == "packet loss"
    assert wp.severity is None

def test_from_value_string_format(self) -> None:
    wp = WeakPoint.from_value("service:crash")
    assert wp.component == "service"
    assert wp.failure_mode == "crash"
    assert wp.severity is None

def test_from_value_string_with_whitespace(self) -> None:
    wp = WeakPoint.from_value("  component  :  failure  ")
    assert wp.component == "component"
    assert wp.failure_mode == "failure"

def test_from_value_string_with_multiple_colons(self) -> None:
    wp = WeakPoint.from_value("db:connection:timeout")
    assert wp.component == "db"
    assert wp.failure_mode == "connection:timeout"

@pytest.mark.parametrize(
    "invalid_input",
    [
        123,
        None,
        [],
        {"component": "test"},  # Missing failure_mode
        {"failure_mode": "test"},  # Missing component
        {"component": "", "failure_mode": "test"},  # Empty component
        {"component": "test", "failure_mode": ""},  # Empty failure_mode
        {"component": 123, "failure_mode": "test"},  # Non-string component
        {"component": "test", "failure_mode": 123},  # Non-string failure_mode
        {"component": "test", "failure_mode": "fail", "severity": 123},  # Non-string severity
        "no_colon",  # String without colon
        ":failure",  # Empty component in string
        "component:",  # Empty failure_mode in string
    ],
)
def test_from_value_rejects_invalid_input(self, invalid_input: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        WeakPoint.from_value(invalid_input)

def test_immutability(self) -> None:
    wp = WeakPoint(component="test", failure_mode="fail")
    with pytest.raises(AttributeError):
        wp.component = "modified"  # type: ignore

def test_equality(self) -> None:
    wp1 = WeakPoint(component="db", failure_mode="lag", severity="high")
    wp2 = WeakPoint(component="db", failure_mode="lag", severity="high")
    wp3 = WeakPoint(component="db", failure_mode="lag", severity="low")

    assert wp1 == wp2
    assert wp1 != wp3

def test_hashability(self) -> None:
    wp1 = WeakPoint(component="cache", failure_mode="miss")
    wp2 = WeakPoint(component="cache", failure_mode="miss")

    # Can be used in sets and dicts
    weak_points = {wp1, wp2}
    assert len(weak_points) == 1

    mapping = {wp1: "value"}
    assert mapping[wp2] == "value"
```

# ============================================================================

# Enhanced TemporalAntibody Tests

# ============================================================================

class TestTemporalAntibody:
“”“Tests for TemporalAntibody validation and behavior.”””

```
def test_construction_valid(self) -> None:
    wp = WeakPoint(component="api", failure_mode="timeout")
    antibody = TemporalAntibody(target=wp, countermeasure="Add circuit breaker")

    assert antibody.target == wp
    assert antibody.countermeasure == "Add circuit breaker"

def test_construction_rejects_non_weak_point_target(self) -> None:
    with pytest.raises(TypeError, match="Target must be a WeakPoint"):
        TemporalAntibody(target="not a weak point", countermeasure="test")  # type: ignore

def test_construction_rejects_empty_countermeasure(self) -> None:
    wp = WeakPoint(component="test", failure_mode="fail")
    with pytest.raises(ValueError, match="non-empty string"):
        TemporalAntibody(target=wp, countermeasure="")

def test_construction_rejects_non_string_countermeasure(self) -> None:
    wp = WeakPoint(component="test", failure_mode="fail")
    with pytest.raises(ValueError):
        TemporalAntibody(target=wp, countermeasure=123)  # type: ignore

def test_immutability(self) -> None:
    wp = WeakPoint(component="test", failure_mode="fail")
    antibody = TemporalAntibody(target=wp, countermeasure="fix it")

    with pytest.raises(AttributeError):
        antibody.countermeasure = "modified"  # type: ignore
```

# ============================================================================

# Enhanced ProbabilityWeaver Tests

# ============================================================================

class TestProbabilityWeaver:
“”“Tests for probability tracking functionality.”””

```
def test_initial_state(self) -> None:
    weaver = ProbabilityWeaver()
    wp = WeakPoint(component="test", failure_mode="fail")

    assert weaver.probability_of(wp) == 0.0

def test_set_and_get_probability(self) -> None:
    weaver = ProbabilityWeaver()
    wp = WeakPoint(component="cache", failure_mode="eviction")

    weaver.set_probability(wp, 0.75)
    assert weaver.probability_of(wp) == 0.75

def test_update_probability(self) -> None:
    weaver = ProbabilityWeaver()
    wp = WeakPoint(component="db", failure_mode="lock")

    weaver.set_probability(wp, 0.5)
    assert weaver.probability_of(wp) == 0.5

    weaver.set_probability(wp, 0.8)
    assert weaver.probability_of(wp) == 0.8

def test_multiple_weak_points(self) -> None:
    weaver = ProbabilityWeaver()
    wp1 = WeakPoint(component="api", failure_mode="timeout")
    wp2 = WeakPoint(component="db", failure_mode="lag")

    weaver.set_probability(wp1, 0.3)
    weaver.set_probability(wp2, 0.7)

    assert weaver.probability_of(wp1) == 0.3
    assert weaver.probability_of(wp2) == 0.7

def test_clear_probabilities(self) -> None:
    weaver = ProbabilityWeaver()
    wp = WeakPoint(component="test", failure_mode="fail")

    weaver.set_probability(wp, 0.9)
    assert weaver.probability_of(wp) == 0.9

    weaver.clear()
    assert weaver.probability_of(wp) == 0.0

@pytest.mark.parametrize("invalid_prob", [-0.1, 1.1, 2.0, -1.0])
def test_rejects_invalid_probability_range(self, invalid_prob: float) -> None:
    weaver = ProbabilityWeaver()
    wp = WeakPoint(component="test", failure_mode="fail")

    with pytest.raises(ValueError, match="between 0 and 1"):
        weaver.set_probability(wp, invalid_prob)

def test_boundary_probabilities(self) -> None:
    weaver = ProbabilityWeaver()
    wp = WeakPoint(component="test", failure_mode="fail")

    weaver.set_probability(wp, 0.0)
    assert weaver.probability_of(wp) == 0.0

    weaver.set_probability(wp, 1.0)
    assert weaver.probability_of(wp) == 1.0
```

# ============================================================================

# Enhanced CausalityEngine Tests

# ============================================================================

class TestCausalityEngine:
“”“Tests for failure origin tracing and normalization.”””

```
def test_empty_weak_points_list(self) -> None:
    engine = CausalityEngine()
    result = engine.trace_failure_origins({"weak_points": []})
    assert result == []

def test_single_dict_weak_point(self) -> None:
    engine = CausalityEngine()
    result = engine.trace_failure_origins(
        {
            "weak_points": [
                {"component": "api", "failure_mode": "crash", "severity": "high"}
            ]
        }
    )

    assert len(result) == 1
    assert result[0].component == "api"
    assert result[0].failure_mode == "crash"
    assert result[0].severity == "high"

def test_single_string_weak_point(self) -> None:
    engine = CausalityEngine()
    result = engine.trace_failure_origins({"weak_points": ["db:timeout"]})

    assert len(result) == 1
    assert result[0].component == "db"
    assert result[0].failure_mode == "timeout"
    assert result[0].severity is None

def test_single_weak_point_object(self) -> None:
    engine = CausalityEngine()
    wp = WeakPoint(component="cache", failure_mode="miss")
    result = engine.trace_failure_origins({"weak_points": [wp]})

    assert len(result) == 1
    assert result[0] == wp

def test_missing_weak_points_key(self) -> None:
    engine = CausalityEngine()
    with pytest.raises(KeyError, match="weak_points"):
        engine.trace_failure_origins({})

def test_weak_points_not_a_list(self) -> None:
    engine = CausalityEngine()
    with pytest.raises(TypeError, match="must be a list"):
        engine.trace_failure_origins({"weak_points": "not a list"})

def test_invalid_weak_point_in_list(self) -> None:
    engine = CausalityEngine()
    with pytest.raises(ValueError, match="Failed to parse"):
        engine.trace_failure_origins({"weak_points": [123]})
```

# ============================================================================

# Enhanced PropheticOrchestrator Tests

# ============================================================================

class TestPropheticOrchestrator:
“”“Comprehensive tests for the main orchestrator.”””

```
def test_initial_state(self) -> None:
    orchestrator = PropheticOrchestrator()
    assert orchestrator.deployed_antibodies == []
    assert isinstance(orchestrator.causality_engine, CausalityEngine)
    assert isinstance(orchestrator.probability_weaver, ProbabilityWeaver)

def test_prevent_single_failure(self) -> None:
    orchestrator = PropheticOrchestrator()
    predicted_failure = {
        "weak_points": [{"component": "api", "failure_mode": "timeout"}]
    }

    antibodies = orchestrator.prevent_future_failure(predicted_failure)

    assert len(antibodies) == 1
    assert isinstance(antibodies[0], TemporalAntibody)
    assert antibodies[0].target.component == "api"
    assert "api" in antibodies[0].countermeasure

def test_prevent_multiple_failures(self) -> None:
    orchestrator = PropheticOrchestrator()
    predicted_failure = {
        "weak_points": [
            "db:lag",
            {"component": "cache", "failure_mode": "eviction"},
            WeakPoint(component="api", failure_mode="crash", severity="critical"),
        ]
    }

    antibodies = orchestrator.prevent_future_failure(predicted_failure)

    assert len(antibodies) == 3
    assert all(isinstance(ab, TemporalAntibody) for ab in antibodies)

def test_countermeasure_includes_component_name(self) -> None:
    orchestrator = PropheticOrchestrator()
    predicted_failure = {
        "weak_points": [
            {"component": "payment_processor", "failure_mode": "transaction failure"}
        ]
    }

    antibodies = orchestrator.prevent_future_failure(predicted_failure)

    assert "payment_processor" in antibodies[0].countermeasure

def test_countermeasure_varies_by_severity(self) -> None:
    orchestrator = PropheticOrchestrator()
    predicted_failure = {
        "weak_points": [
            {"component": "test1", "failure_mode": "fail", "severity": "low"},
            {"component": "test2", "failure_mode": "fail", "severity": "medium"},
            {"component": "test3", "failure_mode": "fail", "severity": "high"},
            {"component": "test4", "failure_mode": "fail", "severity": "critical"},
        ]
    }

    antibodies = orchestrator.prevent_future_failure(predicted_failure)

    # Different severities should produce different countermeasure prefixes
    countermeasures = [ab.countermeasure for ab in antibodies]
    assert len(set(countermeasures)) > 1  # Not all identical

def test_probability_set_to_high_value(self) -> None:
    orchestrator = PropheticOrchestrator()
    predicted_failure = {"weak_points": ["api:timeout"]}

    antibodies = orchestrator.prevent_future_failure(predicted_failure)

    for antibody in antibodies:
        prob = orchestrator.probability_weaver.probability_of(antibody.target)
        assert prob == 0.9999

def test_deployed_antibodies_tracked(self) -> None:
    orchestrator = PropheticOrchestrator()
    predicted_failure = {"weak_points": ["db:lag", "cache:miss"]}

    antibodies = orchestrator.prevent_future_failure(predicted_failure)

    assert orchestrator.deployed_antibodies == antibodies
    assert len(orchestrator.deployed_antibodies) == 2

def test_subsequent_deployment_replaces_previous(self) -> None:
    orchestrator = PropheticOrchestrator()

    # First deployment
    antibodies1 = orchestrator.prevent_future_failure(
        {"weak_points": ["api:timeout"]}
    )
    assert len(orchestrator.deployed_antibodies) == 1

    # Second deployment
    antibodies2 = orchestrator.prevent_future_failure(
        {"weak_points": ["db:lag", "cache:miss"]}
    )
    assert len(orchestrator.deployed_antibodies) == 2
    assert orchestrator.deployed_antibodies == antibodies2
    assert orchestrator.deployed_antibodies != antibodies1

def test_reset_clears_state(self) -> None:
    orchestrator = PropheticOrchestrator()
    orchestrator.prevent_future_failure({"weak_points": ["api:timeout"]})

    assert len(orchestrator.deployed_antibodies) > 0

    orchestrator.reset()

    assert orchestrator.deployed_antibodies == []
    # Check that probabilities are also cleared
    wp = WeakPoint(component="api", failure_mode="timeout")
    assert orchestrator.probability_weaver.probability_of(wp) == 0.0

def test_get_deployment_summary_empty(self) -> None:
    orchestrator = PropheticOrchestrator()
    summary = orchestrator.get_deployment_summary()

    assert summary["total_deployed"] == 0
    assert summary["by_severity"] == {}
    assert summary["by_component"] == {}

def test_get_deployment_summary_with_antibodies(self) -> None:
    orchestrator = PropheticOrchestrator()
```