# file: filter_node.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping

__all__ = ["FilterNode", "RawValue", "raw", "escape_filter_value"]

# Names are strict: alphanumeric + underscore.
_OPERATION_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_PARAM_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")

# Filtergraph separators/syntax chars that must be escaped inside option values.
# Why: unescaped occurrences can split nodes/options or confuse the parser.
_ESCAPE_CHARS = {":", "=", ",", ";", "[", "]", "\\"}


def escape_filter_value(text: str) -> str:
    """Escape FFmpeg filtergraph-special characters inside a value."""
    # Only escape characters with special meaning in filtergraphs.
    # Do not over-escape to keep output readable.
    out_chars = []
    for ch in text:
        if ch in _ESCAPE_CHARS:
            out_chars.append("\\" + ch)
        else:
            out_chars.append(ch)
    return "".join(out_chars)


@dataclass(frozen=True)
class RawValue:
    """
    Marker type to opt-out of escaping. Use when you provide a value that is
    already valid for FFmpeg (e.g., expressions like 'N/FRAME_RATE').
    """
    text: str


def raw(value: Any) -> RawValue:
    """Wrap a value to mark it as pre-escaped/raw for FFmpeg."""
    if isinstance(value, RawValue):
        return value
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str):
        value = _coerce_to_str(value)  # still validate/coerce non-strs
    return RawValue(value)


def _coerce_to_str(value: Any) -> str:
    """Coerce supported types into a string; no escaping here."""
    if isinstance(value, bool):
        return "1" if value else "0"

    # bool is a subclass of int; must come first
    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Float parameters must be finite numbers")
        # Compact fixed-precision; avoids scientific notation surprises.
        s = ("%0.6f" % value).rstrip("0").rstrip(".")
        return s or "0"

    if isinstance(value, Enum):
        # Recurse once on the underlying enum value
        return _coerce_to_str(value.value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, str):
        if value == "":
            raise ValueError("String parameters cannot be empty")
        if "\n" in value or "\r" in value:
            raise ValueError("String parameters cannot contain newlines")
        return value

    raise TypeError("Filter parameter values must be strings, numbers, booleans, Enums, or Paths")


@dataclass(frozen=True)
class FilterNode:
    """Represent a single FFmpeg filter node with validated, escaped syntax."""

    operation: str
    parameters: Dict[str, str]  # stored as sanitized + escaped strings

    def __init__(self, operation: str, parameters: Mapping[str, Any] | None = None):
        if not isinstance(operation, str) or not operation:
            raise TypeError("operation must be a non-empty string")
        if not _OPERATION_PATTERN.match(operation):
            raise ValueError("operation must contain only alphanumeric characters and underscores")

        object.__setattr__(self, "operation", operation)
        sanitized = self._sanitize(parameters or {})
        object.__setattr__(self, "parameters", sanitized)

    @staticmethod
    def _sanitize(parameters: Mapping[str, Any]) -> Dict[str, str]:
        """Return a sanitized, type-checked, escaped copy of the provided parameters."""
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")

        sanitized: Dict[str, str] = {}
        for key, value in parameters.items():
            if value is None:
                # Omit unset parameters to let FFmpeg defaults apply.
                continue

            if not isinstance(key, str) or not key:
                raise TypeError("Parameter names must be non-empty strings")
            if not _PARAM_PATTERN.match(key):
                raise ValueError(
                    "Parameter names must contain only alphanumeric characters and underscores"
                )

            if isinstance(value, RawValue):
                s = value.text
                # Still enforce empty/newline checks for safety.
                if s == "":
                    raise ValueError("String parameters cannot be empty")
                if "\n" in s or "\r" in s:
                    raise ValueError("String parameters cannot contain newlines")
            else:
                s = _coerce_to_str(value)
                s = escape_filter_value(s)

            sanitized[key] = s

        # Deterministic output
        return dict(sorted(sanitized.items(), key=lambda x: x[0]))

    def compile(self) -> str:
        """Compile the node into FFmpeg filter syntax."""
        if not self.parameters:
            return self.operation
        parts = [f"{name}={value}" for name, value in self.parameters.items()]
        return f"{self.operation}=" + ":".join(parts)

    def __str__(self) -> str:  # makes printing/debugging nicer
        return self.compile()

    def with_params(self, more: Mapping[str, Any]) -> "FilterNode":
        """Return a new node with extra/overridden parameters (immutable)."""
        # Merge, with 'more' overriding collisions.
        merged: Dict[str, Any] = {**self.parameters}
        # Self.parameters already escaped; feeding them back through sanitize would double-escape.
        # So we rebuild using the raw textual values as RawValue to prevent re-escaping.
        merged_raw = {k: RawValue(v) for k, v in merged.items()}
        merged_raw.update(more)
        return FilterNode(self.operation, merged_raw)

    def as_mapping(self) -> Dict[str, str]:
        """Return a shallow copy of sanitized parameters."""
        return dict(self.parameters)


# --------------------------- tests (pytest) ---------------------------

# file: test_filter_node.py
def test_compile_no_params():
    node = FilterNode("scale")
    assert node.compile() == "scale"

def test_operation_validation():
    import pytest
    with pytest.raises(ValueError):
        FilterNode("bad-op")
    with pytest.raises(TypeError):
        FilterNode("")

def test_param_name_validation():
    import pytest
    with pytest.raises(ValueError):
        FilterNode("scale", {"w:h": 1280})
    with pytest.raises(TypeError):
        FilterNode("scale", {None: 720})

def test_omit_none():
    node = FilterNode("scale", {"w": 1280, "h": None})
    assert node.compile() == "scale=w=1280"

def test_value_coercions(tmp_path: Path):
    p = tmp_path / "a b:c.txt"
    p.write_text("x")
    node = FilterNode(
        "test",
        {
            "b": True,
            "i": 7,
            "f": 3.1415926535,
            "p": p,
            "s": "ok",
        },
    )
    compiled = node.compile()
    # float trimmed, path escaped (':' becomes '\:')
    assert "b=1" in compiled
    assert "i=7" in compiled
    assert "f=3.141593" in compiled
    assert "p=" in compiled and r"\:" in compiled
    assert "s=ok" in compiled

def test_float_nonfinite_raises():
    import pytest, math
    with pytest.raises(ValueError):
        FilterNode("x", {"f": math.inf})

def test_string_constraints():
    import pytest
    with pytest.raises(ValueError):
        FilterNode("x", {"s": ""})
    with pytest.raises(ValueError):
        FilterNode("x", {"s": "bad\nnewline"})

def test_escaping():
    node = FilterNode("drawtext", {"text": "A:B=C, [x]; end\\slash"})
    # Expect colons, equals, comma, semicolon, brackets, backslash escaped.
    assert node.compile() == "drawtext=text=A\\:B\\=C\\, \\[x\\]\\; end\\\\slash"

def test_raw_bypass():
    node = FilterNode("setpts", {"expr": RawValue("N/FRAME_RATE")})
    assert node.compile() == "setpts=expr=N/FRAME_RATE"

def test_deterministic_sort():
    node = FilterNode("a", {"b": 1, "a": 2})
    assert node.compile() == "a=a=2:b=1"

def test_with_params_immutable_merge():
    base = FilterNode("scale", {"w": 1280})
    nxt = base.with_params({"h": 720})
    assert str(base) == "scale=w=1280"
    assert str(nxt) == "scale=h=720:w=1280"
    # Override existing
    nxt2 = nxt.with_params({"w": 1920})
    assert str(nxt2) == "scale=h=720:w=1920"

# Optional: run a quick smoke when invoked directly.
if __name__ == "__main__":  # pragma: no cover
    n = FilterNode("drawtext", {"text": "hello:world=ok, [demo]"})
    print(n)  # drawtext=text=hello\:world\=ok\, \[demo\]