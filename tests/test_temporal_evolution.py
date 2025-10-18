“”“Comprehensive test suite for Temporal Evolution Roadmap.

This test suite covers:

- Basic parsing and serialization
- Edge cases and boundary conditions
- Comprehensive validation
- Markdown output formatting
- Error messages and exception types
  “””

from **future** import annotations

import pytest

from temporal_evolution import (
Directive,
Discipline,
TemporalEvolutionRoadmap,
)

# ============================================================================

# Test Fixtures

# ============================================================================

def _sample_payload() -> dict[str, object]:
“”“Sample payload matching the original test.”””
return {
“temporal_evolution”: {
“predictive_refactoring”: [
{
“Monitor”: (
“Industry trend APIs, competitor repositories, academic papers”
)
},
{“Predict”: “Future architectural needs using transformer models”},
{“Pre-adapt”: “Generate compatibility layers before standards emerge”},
],
“quantum_branching”: [
“Maintain parallel reality branches of the codebase”,
“Each branch optimized for different future scenarios”,
“Collapse to optimal branch when future becomes present”,
],
}
}

def _minimal_payload() -> dict[str, object]:
“”“Minimal valid payload.”””
return {“temporal_evolution”: {“discipline”: [“directive”]}}

# ============================================================================

# Directive Tests

# ============================================================================

class TestDirective:
“”“Tests for the Directive class.”””

```
def test_simple_directive_from_string(self) -> None:
    directive = Directive.from_value("Do something important")
    assert directive.summary == "Do something important"
    assert directive.detail is None

def test_detailed_directive_from_dict(self) -> None:
    directive = Directive.from_value(
        {"Monitor": "Industry trend APIs, competitor repositories"}
    )
    assert directive.summary == "Monitor"
    assert directive.detail == "Industry trend APIs, competitor repositories"

def test_simple_directive_serialization(self) -> None:
    directive = Directive(summary="Simple task")
    assert directive.serialise() == "Simple task"

def test_detailed_directive_serialization(self) -> None:
    directive = Directive(summary="Monitor", detail="Watch the trends")
    assert directive.serialise() == {"Monitor": "Watch the trends"}

@pytest.mark.parametrize(
    "invalid_value",
    [
        123,
        None,
        [],
        {"key1": "val1", "key2": "val2"},  # Multiple keys
        {},  # Empty dict
        {"": "detail"},  # Empty key
        {"   ": "detail"},  # Whitespace key
        {"key": 123},  # Non-string value
    ],
)
def test_from_value_rejects_invalid_types(self, invalid_value: object) -> None:
    with pytest.raises(TypeError):
        Directive.from_value(invalid_value)

def test_from_value_rejects_empty_string(self) -> None:
    with pytest.raises(TypeError, match="cannot be empty"):
        Directive.from_value("")

def test_from_value_rejects_whitespace_string(self) -> None:
    with pytest.raises(TypeError, match="cannot be empty"):
        Directive.from_value("   ")

def test_unicode_handling(self) -> None:
    directive = Directive.from_value("Implement 🚀 features")
    assert directive.summary == "Implement 🚀 features"

    directive = Directive.from_value({"观察": "监控行业趋势"})
    assert directive.summary == "观察"
    assert directive.detail == "监控行业趋势"
```

# ============================================================================

# Discipline Tests

# ============================================================================

class TestDiscipline:
“”“Tests for the Discipline class.”””

```
def test_from_mapping_simple_directives(self) -> None:
    discipline = Discipline.from_mapping(
        "testing", ["Write tests", "Run tests", "Fix failures"]
    )
    assert discipline.name == "testing"
    assert len(discipline.directives) == 3
    assert discipline.directives[0].summary == "Write tests"

def test_from_mapping_mixed_directives(self) -> None:
    discipline = Discipline.from_mapping(
        "mixed",
        [
            "Simple directive",
            {"Complex": "With details"},
            "Another simple one",
        ],
    )
    assert len(discipline.directives) == 3
    assert discipline.directives[0].detail is None
    assert discipline.directives[1].detail == "With details"
    assert discipline.directives[2].detail is None

def test_serialization(self) -> None:
    discipline = Discipline.from_mapping(
        "test", ["Simple", {"Complex": "Details"}]
    )
    serialized = discipline.serialise()
    assert serialized == ["Simple", {"Complex": "Details"}]

def test_to_markdown_simple_directives(self) -> None:
    discipline = Discipline.from_mapping("testing", ["Write tests", "Run tests"])
    markdown = discipline.to_markdown()

    assert "### Testing" in markdown
    assert "- Write tests" in markdown
    assert "- Run tests" in markdown

def test_to_markdown_detailed_directives(self) -> None:
    discipline = Discipline.from_mapping(
        "monitoring", [{"Monitor": "Watch the systems closely"}]
    )
    markdown = discipline.to_markdown()

    assert "### Monitoring" in markdown
    assert "**Monitor**" in markdown
    assert "Watch the systems closely" in markdown

def test_name_formatting(self) -> None:
    discipline = Discipline.from_mapping("predictive_refactoring", [])
    markdown = discipline.to_markdown()
    assert "### Predictive Refactoring" in markdown

@pytest.mark.parametrize(
    "invalid_name",
    [42, None, "", "   ", [], {}],
)
def test_from_mapping_rejects_invalid_names(self, invalid_name: object) -> None:
    with pytest.raises(TypeError):
        Discipline.from_mapping(invalid_name, [])

@pytest.mark.parametrize(
    "invalid_directives",
    [
        "not a list",
        {"dict": "instead"},
        42,
        None,
    ],
)
def test_from_mapping_rejects_invalid_directives_type(
    self, invalid_directives: object
) -> None:
    with pytest.raises(TypeError):
        Discipline.from_mapping("valid_name", invalid_directives)

def test_empty_directives_list_allowed(self) -> None:
    discipline = Discipline.from_mapping("empty", [])
    assert len(discipline.directives) == 0
    assert discipline.serialise() == []
```

# ============================================================================

# TemporalEvolutionRoadmap Tests (Original)

# ============================================================================

def test_from_mapping_parses_nested_temporal_evolution_block() -> None:
“”“Original test case.”””
roadmap = TemporalEvolutionRoadmap.from_mapping(_sample_payload())

```
assert [discipline.name for discipline in roadmap.disciplines] == [
    "predictive_refactoring",
    "quantum_branching",
]

predictive = roadmap.disciplines[0]
assert [directive.summary for directive in predictive.directives] == [
    "Monitor",
    "Predict",
    "Pre-adapt",
]
assert predictive.directives[0].detail.startswith("Industry trend APIs")

quantum = roadmap.disciplines[1]
assert [directive.detail for directive in quantum.directives] == [None, None, None]
assert "parallel reality" in quantum.directives[0].summary

markdown = roadmap.to_markdown()
assert "### Predictive Refactoring" in markdown
assert "**Monitor**" in markdown
assert "- Maintain parallel reality branches" in markdown
```

@pytest.mark.parametrize(
“invalid_directives”,
[
“plain string”,
{“unexpected”: “mapping”},
[123],
[{“Monitor”: 10}],
[{“Monitor”: “ok”, “Extra”: “nope”}],
[{”   “: “detail”}],
],
)
def test_from_mapping_validates_directive_structure(invalid_directives) -> None:
“”“Original validation test.”””
payload = {“temporal_evolution”: {“predictive_refactoring”: invalid_directives}}

```
with pytest.raises(TypeError):
    TemporalEvolutionRoadmap.from_mapping(payload)
```

def test_serialise_round_trip() -> None:
“”“Original round-trip test.”””
payload = _sample_payload()
roadmap = TemporalEvolutionRoadmap.from_mapping(payload)

```
assert roadmap.serialise() == payload["temporal_evolution"]
```

@pytest.mark.parametrize(
“invalid_name”,
[42, “   “],
)
def test_from_mapping_validates_discipline_names(invalid_name) -> None:
“”“Original name validation test.”””
payload = {“temporal_evolution”: {invalid_name: []}}

```
with pytest.raises(TypeError):
    TemporalEvolutionRoadmap.from_mapping(payload)
```

# ============================================================================

# Enhanced TemporalEvolutionRoadmap Tests

# ============================================================================

class TestTemporalEvolutionRoadmap:
“”“Enhanced tests for the complete roadmap.”””

```
def test_minimal_valid_payload(self) -> None:
    roadmap = TemporalEvolutionRoadmap.from_mapping(_minimal_payload())
    assert len(roadmap.disciplines) == 1
    assert roadmap.disciplines[0].name == "discipline"

def test_empty_disciplines_allowed(self) -> None:
    roadmap = TemporalEvolutionRoadmap.from_mapping({"temporal_evolution": {}})
    assert len(roadmap.disciplines) == 0

def test_from_mapping_rejects_missing_key(self) -> None:
    with pytest.raises(KeyError, match="temporal_evolution"):
        TemporalEvolutionRoadmap.from_mapping({})

def test_from_mapping_rejects_wrong_type_for_key(self) -> None:
    with pytest.raises(TypeError):
        TemporalEvolutionRoadmap.from_mapping({"temporal_evolution": "string"})

def test_from_mapping_rejects_non_dict_input(self) -> None:
    with pytest.raises(TypeError):
        TemporalEvolutionRoadmap.from_mapping("not a dict")

def test_serialization_preserves_order(self) -> None:
    payload = {
        "temporal_evolution": {
            "first": ["a", "b"],
            "second": ["c", "d"],
            "third": ["e", "f"],
        }
    }
    roadmap = TemporalEvolutionRoadmap.from_mapping(payload)
    serialized = roadmap.serialise()

    # Check names are present (dict order preservation in Python 3.7+)
    assert list(serialized.keys()) == ["first", "second", "third"]

def test_to_markdown_structure(self) -> None:
    roadmap = TemporalEvolutionRoadmap.from_mapping(_sample_payload())
    markdown = roadmap.to_markdown()

    # Check overall structure
    assert markdown.startswith("## Temporal Evolution Roadmap")
    assert markdown.count("###") == 2  # Two disciplines
    assert markdown.count("**") >= 6  # Three detailed directives in first discipline

def test_to_markdown_with_empty_roadmap(self) -> None:
    roadmap = TemporalEvolutionRoadmap.from_mapping({"temporal_evolution": {}})
    markdown = roadmap.to_markdown()
    assert markdown == "## Temporal Evolution Roadmap\n"

def test_complex_nested_structure(self) -> None:
    """Test a more complex realistic scenario."""
    payload = {
        "temporal_evolution": {
            "ai_integration": [
                {"Phase 1": "Research current AI capabilities"},
                {"Phase 2": "Design integration architecture"},
                "Implement proof of concept",
                {"Phase 3": "Roll out to production"},
            ],
            "security_hardening": [
                "Conduct security audit",
                "Apply patches",
                {"Monitor": "Set up 24/7 monitoring systems"},
            ],
        }
    }

    roadmap = TemporalEvolutionRoadmap.from_mapping(payload)

    # Verify structure
    assert len(roadmap.disciplines) == 2
    assert roadmap.disciplines[0].name == "ai_integration"
    assert len(roadmap.disciplines[0].directives) == 4
    assert roadmap.disciplines[1].name == "security_hardening"
    assert len(roadmap.disciplines[1].directives) == 3

    # Round-trip
    assert roadmap.serialise() == payload["temporal_evolution"]

    # Markdown
    markdown = roadmap.to_markdown()
    assert "### Ai Integration" in markdown
    assert "### Security Hardening" in markdown
    assert "**Phase 1**" in markdown

def test_unicode_in_full_roadmap(self) -> None:
    """Test Unicode handling throughout the structure."""
    payload = {
        "temporal_evolution": {
            "国际化": ["支持多语言", {"优先级": "高"}],
            "accessibility": ["Support 日本語", "Add ♿ features"],
        }
    }

    roadmap = TemporalEvolutionRoadmap.from_mapping(payload)
    serialized = roadmap.serialise()
    markdown = roadmap.to_markdown()

    assert "国际化" in str(serialized)
    assert "♿" in markdown

def test_very_long_strings(self) -> None:
    """Test handling of very long directive text."""
    long_text = "A" * 10000
    payload = {
        "temporal_evolution": {
            "testing": [long_text, {"Summary": long_text}]
        }
    }

    roadmap = TemporalEvolutionRoadmap.from_mapping(payload)
    assert len(roadmap.disciplines[0].directives[0].summary) == 10000
    assert len(roadmap.disciplines[0].directives[1].detail) == 10000

def test_immutability(self) -> None:
    """Verify that dataclasses are frozen."""
    roadmap = TemporalEvolutionRoadmap.from_mapping(_minimal_payload())

    with pytest.raises(AttributeError):
        roadmap.disciplines = ()  # type: ignore

    with pytest.raises(AttributeError):
        roadmap.disciplines[0].name = "modified"  # type: ignore

    with pytest.raises(AttributeError):
        roadmap.disciplines[0].directives[0].summary = "modified"  # type: ignore
```

# ============================================================================

# Integration Tests

# ============================================================================

class TestIntegration:
“”“End-to-end integration tests.”””

```
def test_parse_serialize_parse_idempotence(self) -> None:
    """Test that parse -> serialize -> parse is idempotent."""
    payload = _sample_payload()

    roadmap1 = TemporalEvolutionRoadmap.from_mapping(payload)
    serialized = {"temporal_evolution": roadmap1.serialise()}
    roadmap2 = TemporalEvolutionRoadmap.from_mapping(serialized)

    # Compare structure
    assert len(roadmap1.disciplines) == len(roadmap2.disciplines)
    for d1, d2 in zip(roadmap1.disciplines, roadmap2.disciplines):
        assert d1.name == d2.name
        assert len(d1.directives) == len(d2.directives)
        for dir1, dir2 in zip(d1.directives, d2.directives):
            assert dir1.summary == dir2.summary
            assert dir1.detail == dir2.detail

def test_markdown_output_comprehensive(self) -> None:
    """Verify markdown output matches expected format exactly."""
    roadmap = TemporalEvolutionRoadmap.from_mapping(_sample_payload())
    markdown = roadmap.to_markdown()

    # Split into lines for detailed verification
    lines = markdown.split("\n")

    # Check header
    assert lines[0] == "## Temporal Evolution Roadmap"
    assert lines[1] == ""

    # Check discipline headers are present
    discipline_headers = [line for line in lines if line.startswith("###")]
    assert len(discipline_headers) == 2
    assert "Predictive Refactoring" in discipline_headers[0]
    assert "Quantum Branching" in discipline_headers[1]

    # Check bullet points for simple directives
    bullet_points = [line for line in lines if line.startswith("- ")]
    assert len(bullet_points) == 3  # Three in quantum_branching

    # Check bold summaries for detailed directives
    bold_items = [line for line in lines if line.startswith("**")]
    assert len(bold_items) == 3  # Three in predictive_refactoring
```