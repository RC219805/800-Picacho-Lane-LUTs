# path: codebase_philosophy_auditor.py
"""Utility for auditing modules against high-level codebase principles.

The :class:`CodebasePhilosophyAuditor` inspects a Python module, extracts
``# Decision:`` annotations, and then applies a set of simple rules derived
from the repository's philosophy guidelines. The goal isn't to perform heavy
static analysis, but to provide lightweight guardrails that surface common
policy violations and highlight where explicit decisions were made to bend the
rules.

Run as a CLI to audit files or directories recursively.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

# -----------------------------
# Decision annotations
# -----------------------------

_DECISION_PATTERN = re.compile(
    r"#\s*Decision\s*:\s*(?P<name>[A-Za-z0-9_\-]+)(?:\s*-\s*(?P<text>.*))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Decision:
    """Represents an explicit decision documented in the source code."""
    name: str
    line: int
    rationale: Optional[str] = None


@dataclass(frozen=True)
class Violation:
    """Information about a principle violation discovered during the audit."""
    principle: str
    message: str
    line: Optional[int] = None
    decision: Optional[Decision] = None


@dataclass
class _AuditContext:
    """Runtime context shared by the auditing rules."""
    source_lines: List[str]
    decisions: List[Decision]
    max_distance: int = 2  # how far a line-local decision can apply

    def global_decision(self, name: str) -> Optional[Decision]:
        name = name.lower()
        for decision in self.decisions:
            if decision.name == name:
                return decision
        return None

    def decision_for_line(self, name: str, line: int) -> Optional[Decision]:
        """Find a nearby decision named *name* within `max_distance` lines above."""
        name = name.lower()
        for decision in reversed(self.decisions):
            if decision.name != name:
                continue
            if 0 <= line - decision.line <= self.max_distance:
                return decision
        return None


def _extract_decisions(source_lines: Iterable[str]) -> List[Decision]:
    decisions: List[Decision] = []
    for index, line in enumerate(source_lines, start=1):
        match = _DECISION_PATTERN.search(line)
        if not match:
            continue
        name = match.group("name").strip().lower()
        rationale = match.group("text")
        if rationale is not None:
            rationale = rationale.strip() or None
        decisions.append(Decision(name=name, line=index, rationale=rationale))
    return decisions


# -----------------------------
# Rules
# -----------------------------

def _check_module_docstring(tree: ast.AST, context: _AuditContext) -> List[Violation]:
    if ast.get_docstring(tree, clean=False) is not None:
        return []
    if context.global_decision("allow_missing_docstring"):
        return []
    return [
        Violation(
            principle="module_docstring",
            message="Module is missing a top-level docstring",
            line=1,
        )
    ]


def _check_public_api_docstrings(tree: ast.Module, context: _AuditContext) -> List[Violation]:
    violations: List[Violation] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            if ast.get_docstring(node, clean=False) is not None:
                continue
            decision = context.decision_for_line("undocumented_public_api", node.lineno)
            if decision:
                continue
            violations.append(
                Violation(
                    principle="public_api_documentation",
                    message=f"Public {type(node).__name__.lower()} '{node.name}' lacks a docstring",
                    line=node.lineno,
                )
            )
    return violations


def _check_wildcard_imports(tree: ast.Module, context: _AuditContext) -> List[Violation]:
    violations: List[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    decision = context.decision_for_line("allow_wildcard_import", node.lineno)
                    if decision:
                        continue
                    violations.append(
                        Violation(
                            principle="no_wildcard_imports",
                            message=f"Wildcard import from '{node.module or ''}' violates import policy",
                            line=node.lineno,
                        )
                    )
    return violations


Rule = Callable[[ast.Module, _AuditContext], List[Violation]]


class CodebasePhilosophyAuditor:
    """Audit Python modules for high-level codebase philosophy violations."""

    def __init__(self, rules: Optional[Iterable[Rule]] = None, *, max_distance: int = 2) -> None:
        self._rules: List[Rule] = (
            list(rules)
            if rules is not None
            else [
                _check_module_docstring,
                _check_public_api_docstrings,
                _check_wildcard_imports,
            ]
        )
        self._max_distance = max_distance

    def audit_source(self, source: str, *, filename: str = "<memory>") -> Tuple[List[Violation], Optional[str]]:
        """Audit from a source string. Returns (violations, error_message)."""
        source_lines = source.splitlines()
        decisions = _extract_decisions(source_lines)
        context = _AuditContext(source_lines=source_lines, decisions=decisions, max_distance=self._max_distance)

        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as exc:
            msg = f"{exc.msg} (line {exc.lineno})"
            return [Violation(principle="syntax_error", message=msg, line=exc.lineno)], msg
        except Exception as exc:
            msg = f"parse_error: {exc}"
            return [Violation(principle="parse_error", message=str(exc))], msg

        violations: List[Violation] = []
        for rule in self._rules:
            violations.extend(rule(tree, context))
        return violations, None

    def audit_module(self, module_path: Path, *, encoding: str = "utf-8") -> Tuple[List[Violation], Optional[str]]:
        """Inspect *module_path* and return (violations, error_message)."""
        try:
            source = module_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            msg = f"decode_error: {exc}"
            return [Violation(principle="decode_error", message=str(exc))], msg
        return self.audit_source(source, filename=str(module_path))


# -----------------------------
# Directory + CLI helpers
# -----------------------------

_DEFAULT_EXCLUDES = (
    ".venv", "venv", ".eggs", "build", "dist", "__pycache__", ".git",
    "*.pyc", "*.pyo", "*.pyd",
    "tests/fixtures/*",
)


def _is_excluded(path: Path, patterns: Sequence[str]) -> bool:
    rel = str(path.as_posix())
    parts = set(path.as_posix().split("/"))
    for pat in patterns:
        if "/" not in pat and pat in parts:
            return True
        if fnmatch.fnmatch(rel, pat):
            return True
    return False


def _iter_py_files(paths: Sequence[Path], excludes: Sequence[str]) -> Iterator[Path]:
    patterns = list(excludes)
    for p in paths:
        if p.is_dir():
            for path in p.rglob("*.py"):
                if _is_excluded(path, patterns):
                    continue
                yield path
        elif p.is_file() and p.suffix == ".py":
            if not _is_excluded(p, patterns):
                yield p


def _format_text(path: Path, v: Violation) -> str:
    loc = f"{path}:{v.line}" if v.line else f"{path}"
    return f"{loc} [{v.principle}] {v.message}"


def _serialize_violation(path: Path, v: Violation) -> dict:
    data = {"file": str(path), "principle": v.principle, "message": v.message}
    if v.line is not None:
        data["line"] = v.line
    if v.decision is not None:
        data["decision"] = asdict(v.decision)
    return data


def audit_paths(
    paths: Sequence[Path],
    *,
    excludes: Sequence[str] = _DEFAULT_EXCLUDES,
    encoding: str = "utf-8",
    max_distance: int = 2,
) -> List[Tuple[Path, List[Violation]]]:
    auditor = CodebasePhilosophyAuditor(max_distance=max_distance)
    results: List[Tuple[Path, List[Violation]]] = []
    for f in _iter_py_files(paths, excludes):
        violations, _err = auditor.audit_module(f, encoding=encoding)
        results.append((f, violations))
    return results


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit Python modules for high-level codebase philosophy violations."
    )
    p.add_argument("targets", nargs="+", type=Path, help="Files or directories to audit")
    p.add_argument("--exclude", action="append", default=[], help="Glob to exclude (repeatable)")
    p.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    p.add_argument("--format", choices=("text", "json"), default="text", help="Output format")
    p.add_argument("--fail-on", default="", help="Comma-separated principles to fail on; use 'all' for any violation")
    p.add_argument("--max-distance", type=int, default=2, help="Max lines a Decision applies to (default: 2)")
    return p.parse_args(argv)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ns = _parse_args(argv)
    excludes = tuple(_DEFAULT_EXCLUDES) + tuple(ns.exclude or ())
    results = audit_paths(ns.targets, excludes=excludes, encoding=ns.encoding, max_distance=ns.max_distance)

    all_violations: List[dict] = []
    fail_set = set()
    if ns.fail_on:
        fail_set = set(x.strip() for x in ns.fail_on.split(",") if x.strip())
    fail_any = "all" in fail_set

    exit_code = 0
    if ns.format == "json":
        for path, vs in results:
            for v in vs:
                all_violations.append(_serialize_violation(path, v))
        print(json.dumps(all_violations, ensure_ascii=False, indent=2))
        if fail_any and all_violations:
            exit_code = 2
        elif fail_set:
            if any(v["principle"] in fail_set for v in all_violations):
                exit_code = 2
        return exit_code

    # text format
    had_output = False
    for path, vs in results:
        for v in vs:
            had_output = True
            print(_format_text(path, v))
    if not had_output:
        print("No violations found.")

    if fail_any and had_output:
        exit_code = 2
    elif fail_set:
        if any(v.principle in fail_set for _p, vs in results for v in vs):
            exit_code = 2
    return exit_code


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()