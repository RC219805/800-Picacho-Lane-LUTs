# file: prophetic_orchestrator.py
"""Futuristic failure prevention utilities for the LUT automation suite.

Playful but practical orchestrator with a small CLI:
  - trace:   normalize predicted failure into weak points
  - prevent: reinforce & generate temporal antibodies
  - snapshot:show probability ledger for provided weak points
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Dict, Any


# ------------------------------ core datatypes ------------------------------

@dataclass(frozen=True, eq=False)
class WeakPoint:
    """Represents a vulnerable component within the causal chain."""
    component: str
    failure_mode: str
    severity: Optional[str] = None
    metadata: Optional[Mapping[str, object]] = None

    def signature(self) -> str:
        """Stable identifier for reporting and hashing."""
        if self.severity:
            return f"{self.component}:{self.failure_mode}:{self.severity}"
        return f"{self.component}:{self.failure_mode}"

    # why: default dataclass hash would include `metadata` (often unhashable).
    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self.signature())

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        return isinstance(other, WeakPoint) and self.signature() == other.signature()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "failure_mode": self.failure_mode,
            "severity": self.severity,
            "metadata": dict(self.metadata) if self.metadata else None,
            "signature": self.signature(),
        }


@dataclass(frozen=True)
class TemporalAntibody:
    """Represents a preventative measure aimed at a specific weak point."""
    target: WeakPoint
    countermeasure: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "countermeasure": self.countermeasure,
            "confidence": self.confidence,
        }


# ------------------------------ core engines --------------------------------

class CausalityEngine:
    """Trace the underlying causes of a predicted failure."""

    def trace_failure_origins(self, predicted_failure: object) -> List[WeakPoint]:
        """Normalize *predicted_failure* into a list of weak points."""
        weak_entries: Iterable[object]

        if isinstance(predicted_failure, Mapping):
            weak_entries = (
                predicted_failure.get("weak_points")
                or predicted_failure.get("causes")
                or ()
            )
        elif isinstance(predicted_failure, Sequence) and not isinstance(
            predicted_failure, (str, bytes, bytearray)
        ):
            weak_entries = predicted_failure
        else:
            weak_entries = [predicted_failure]

        normalized: List[WeakPoint] = [self._normalize_entry(entry) for entry in weak_entries]
        return normalized

    def _normalize_entry(self, entry: object) -> WeakPoint:
        if isinstance(entry, WeakPoint):
            return entry

        if isinstance(entry, Mapping):
            component = str(
                entry.get("component")
                or entry.get("system")
                or entry.get("target")
                or entry.get("name")
                or "unknown"
            ).strip()
            failure_mode = str(
                entry.get("failure_mode")
                or entry.get("issue")
                or entry.get("mode")
                or entry.get("description")
                or entry.get("reason")
                or "unspecified"
            ).strip()
            severity = entry.get("severity") or entry.get("criticality")
            metadata_keys = {
                "component", "system", "target", "name",
                "failure_mode", "issue", "mode", "description", "reason",
                "severity", "criticality",
            }
            metadata: Dict[str, object] = {
                str(k): v for k, v in entry.items() if k not in metadata_keys
            }
            return WeakPoint(
                component=component or "unknown",
                failure_mode=failure_mode or "unspecified",
                severity=str(severity) if severity is not None else None,
                metadata=metadata or None,
            )

        if isinstance(entry, str):
            component, _, failure_mode = entry.partition(":")
            component = component.strip() or "unknown"
            failure_mode = failure_mode.strip() or "unspecified"
            return WeakPoint(component=component, failure_mode=failure_mode)

        return WeakPoint(component="unknown", failure_mode=str(entry))


class QuantumProbabilityField:
    """Maintain success probabilities for each weak point."""

    def __init__(self) -> None:
        self._branch_probabilities: MutableMapping[WeakPoint, float] = {}

    def strengthen_reality_branch(
        self, weak_point: WeakPoint, *, success_probability: float
    ) -> float:
        """Record a high-confidence outcome for *weak_point*."""
        if not 0.0 <= success_probability <= 1.0:
            raise ValueError("success_probability must be between 0 and 1")
        current = self._branch_probabilities.get(weak_point, 0.0)
        updated = max(current, float(success_probability))
        self._branch_probabilities[weak_point] = updated
        return updated

    def probability_of(self, weak_point: WeakPoint) -> float:
        return self._branch_probabilities.get(weak_point, 0.0)

    def snapshot(self) -> Mapping[WeakPoint, float]:
        return dict(self._branch_probabilities)

    def snapshot_serializable(self) -> Dict[str, float]:
        """Signature→probability mapping for JSON."""
        return {wp.signature(): p for wp, p in self._branch_probabilities.items()}


class PropheticOrchestrator:
    """Orchestrate proactive fixes for predicted failures."""

    def __init__(
        self,
        timeline_analyzer: Optional[CausalityEngine] = None,
        probability_weaver: Optional[QuantumProbabilityField] = None,
    ) -> None:
        self.timeline_analyzer = timeline_analyzer or CausalityEngine()
        self.probability_weaver = probability_weaver or QuantumProbabilityField()
        self._deployed_antibodies: List[TemporalAntibody] = []

    def prevent_future_failure(
        self, predicted_failure: object
    ) -> List[TemporalAntibody]:
        """Neutralize *predicted_failure* before it manifests."""
        causal_chain = self.timeline_analyzer.trace_failure_origins(predicted_failure)
        for weak_point in causal_chain:
            self.probability_weaver.strengthen_reality_branch(
                weak_point, success_probability=0.9999
            )
        antibodies = self.generate_anti_patterns(predicted_failure, causal_chain)
        self.deploy_temporal_antibodies(antibodies)
        return antibodies

    def generate_anti_patterns(
        self,
        predicted_failure: object,
        causal_chain: Optional[Sequence[WeakPoint]] = None,
    ) -> List[TemporalAntibody]:
        """Create actionable countermeasures for each weak point."""
        if causal_chain is None:
            causal_chain = self.timeline_analyzer.trace_failure_origins(predicted_failure)

        antibodies: List[TemporalAntibody] = []
        for weak_point in causal_chain:
            severity = (weak_point.severity or "medium").lower()
            if severity in {"critical", "high"}:
                confidence = 0.995
            elif severity in {"low", "minor"}:
                confidence = 0.9
            else:
                confidence = 0.96

            countermeasure = (
                f"Install anticipatory guardrails for {weak_point.component} to neutralize "
                f"{weak_point.failure_mode}."
            )
            antibodies.append(
                TemporalAntibody(
                    target=weak_point,
                    countermeasure=countermeasure,
                    confidence=confidence,
                )
            )
        return antibodies

    def deploy_temporal_antibodies(
        self, antibodies: Iterable[TemporalAntibody]
    ) -> None:
        self._deployed_antibodies.extend(antibodies)

    @property
    def deployed_antibodies(self) -> List[TemporalAntibody]:
        return list(self._deployed_antibodies)


__all__ = [
    "WeakPoint",
    "TemporalAntibody",
    "CausalityEngine",
    "QuantumProbabilityField",
    "PropheticOrchestrator",
]


# ---------------------------------- CLI ------------------------------------

def _read_json_arg(path: str | None) -> Any:
    if not path:
        return None
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cli_trace(obj: Any) -> int:
    eng = CausalityEngine()
    wps = eng.trace_failure_origins(obj)
    out = {"ok": True, "count": len(wps), "weak_points": [wp.to_dict() for wp in wps]}
    print(json.dumps(out, indent=2))
    return 0


def _cli_prevent(obj: Any) -> int:
    orch = PropheticOrchestrator()
    antibodies = orch.prevent_future_failure(obj)
    out = {
        "ok": True,
        "deployed": len(antibodies),
        "antibodies": [a.to_dict() for a in antibodies],
        "probabilities": orch.probability_weaver.snapshot_serializable(),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cli_snapshot(obj: Any) -> int:
    """Accepts the same structures as `trace` (or a list of WeakPoint-like dicts)."""
    eng = CausalityEngine()
    field = QuantumProbabilityField()
    for wp in eng.trace_failure_origins(obj):
        field.strengthen_reality_branch(wp, success_probability=field.probability_of(wp))
    print(json.dumps({"ok": True, "probabilities": field.snapshot_serializable()}, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="prophetic", description="Prophetic Orchestrator CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--json", help="Path to JSON input or '-' for stdin.")
        g.add_argument("--string", help="Quick string input like 'db:timeout' or plain text.")

    sp_trace = sub.add_parser("trace", help="Normalize predicted failure → weak points.")
    add_common(sp_trace)

    sp_prev = sub.add_parser("prevent", help="Reinforce & emit antibodies.")
    add_common(sp_prev)

    sp_snap = sub.add_parser("snapshot", help="Emit a probability ledger for provided weak points.")
    add_common(sp_snap)

    args = p.parse_args(argv)
    obj: Any = _read_json_arg(args.json) if args.json is not None else (args.string or "")

    if args.cmd == "trace":
        return _cli_trace(obj)
    if args.cmd == "prevent":
        return _cli_prevent(obj)
    if args.cmd == "snapshot":
        return _cli_snapshot(obj)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())