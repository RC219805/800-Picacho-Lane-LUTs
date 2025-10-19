# file: codebase_philosophy_auditor.py
from __future__ import annotations

"""
Lightweight auditor exposing:
- `Violation`: sortable, frozen dataclass with stable dict/str forms.
- `CodebasePhilosophyAuditor`: stdlib-only scanner returning violations.

Rules (regex-based, intentionally simple for CI speed):
- imports:no-wildcard — forbid `from x import *` (error)
- style:no-hard-tabs — discourage tabs (warning)
- hygiene:todo-leftover — flag TODO/FIXME (info)
- logging:no-print — flag print() in non-tests (info)
- design:long-function — flag functions > N lines (warning)
"""

import fnmatch
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

Severity = str  # "info" | "warning" | "error"
SEVERITY_ORDER: Dict[Severity, int] = {"info": 0, "warning": 1, "error": 2}


@dataclass(order=True, frozen=True)
class Violation:
    """Immutable violation record; ordering sorts by severity→rule→path→line.

    Why frozen: prevents accidental mutation when aggregating or caching.
    """

    sort_index: Tuple[int, str, str, int] = field(init=False, repr=False)

    rule: str
    message: str
    path: str
    line: Optional[int] = None
    severity: Severity = "warning"
    hint: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sev_rank = SEVERITY_ORDER.get(self.severity, 1)
        ln = self.line if isinstance(self.line, int) and self.line is not None and self.line >= 0 else 0
        object.__setattr__(self, "sort_index", (sev_rank, self.rule or "", self.path or "", ln))

    def to_dict(self) -> Dict[str, Any]:
        """Stable dict representation for JSON/IPC."""
        return {
            "rule": self.rule,
            "message": self.message,
            "path": self.path,
            "line": self.line,
            "severity": self.severity,
            "hint": self.hint,
            "meta": dict(self.meta) if self.meta else {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Violation":
        """Inverse of to_dict()."""
        return cls(
            rule=str(data.get("rule", "")),
            message=str(data.get("message", "")),
            path=str(data.get("path", "")),
            line=(int(data["line"]) if data.get("line") is not None else None),
            severity=str(data.get("severity", "warning")),
            hint=(str(data["hint"]) if data.get("hint") is not None else None),
            meta=dict(data.get("meta", {})),
        )

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line is not None else self.path
        s = f"[{self.severity.upper()}] {self.rule} @ {loc} – {self.message}"
        return s + (f" (hint: {self.hint})" if self.hint else "")


class CodebasePhilosophyAuditor:
    """Stdlib-only auditor with conservative, fast heuristics."""

    DEFAULT_INCLUDE: Tuple[str, ...] = ("**/*.py",)
    DEFAULT_IGNORE: Tuple[str, ...] = (
        ".git/**",
        ".hg/**",
        ".svn/**",
        ".venv/**",
        "venv/**",
        "__pycache__/**",
        "site-packages/**",
        "dist/**",
        "build/**",
        "node_modules/**",
    )

    _RE_WILDCARD_IMPORT = re.compile(r"^\s*from\s+[A-Za-z0-9_.]+\s+import\s+\*\s*(#.*)?$", re.MULTILINE)
    _RE_HAS_TAB = re.compile(r"\t")
    _RE_TODO = re.compile(r"#\s*(TODO|FIXME)\b", re.IGNORECASE)
    _RE_PRINT = re.compile(r"(?<!\w)print\(", re.MULTILINE)
    _RE_DEF = re.compile(r"^(\s*)def\s+\w+\s*\(", re.MULTILINE)

    def __init__(
        self,
        root: Optional[os.PathLike[str] | str] = None,
        *,
        include_globs: Sequence[str] | None = None,
        ignore_globs: Sequence[str] | None = None,
        max_files: Optional[int] = 5000,
        long_function_limit: int = 120,
    ) -> None:
        self.root = Path(root) if root is not None else Path.cwd()
        self.include_globs: Tuple[str, ...] = tuple(include_globs or self.DEFAULT_INCLUDE)
        self.ignore_globs: Tuple[str, ...] = tuple(ignore_globs or self.DEFAULT_IGNORE)
        self.max_files = max_files
        self.long_function_limit = int(long_function_limit)

    # ---------- public API ----------

    def scan(self) -> List[Violation]:
        """Walk files and collect violations. Never raises on unreadable files."""
        violations: List[Violation] = []
        for path in self.iter_python_files():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                violations.append(
                    Violation(
                        rule="auditor:file-unreadable",
                        message="Could not read file (permission/encoding).",
                        path=self._relpath(path),
                        line=None,
                        severity="warning",
                    )
                )
                continue
            violations.extend(self.scan_text(path, text))
        return sorted(violations)

    def iter_python_files(self) -> Iterator[Path]:
        """Yield .py files honoring include/ignore and max_files cap."""
        yielded = 0
        for inc in self.include_globs:
            for path in self.root.glob(inc):
                if not path.is_file():
                    continue
                if self._should_ignore(path):
                    continue
                yield path
                yielded += 1
                if self.max_files is not None and yielded >= self.max_files:
                    return

    def scan_text(self, path: os.PathLike[str] | str, text: str) -> List[Violation]:
        """Run all rules on a single file's text."""
        rel = self._relpath(Path(path))
        is_test_file = "test" in Path(rel).name.lower() or "/tests/" in rel.replace("\\", "/")

        out: List[Violation] = []

        for m in self._RE_WILDCARD_IMPORT.finditer(text):
            out.append(
                Violation(
                    rule="imports:no-wildcard",
                    message="Avoid 'from x import *'.",
                    path=rel,
                    line=self._line_no(text, m.start()),
                    severity="error",
                    hint="Import explicit names.",
                )
            )

        # tabs
        for lineno, line in enumerate(text.splitlines(), start=1):
            if self._RE_HAS_TAB.search(line):
                out.append(
                    Violation(
                        rule="style:no-hard-tabs",
                        message="Use spaces only.",
                        path=rel,
                        line=lineno,
                        severity="warning",
                        hint="Editor: expand tabs to spaces.",
                    )
                )

        # TODO/FIXME
        for m in self._RE_TODO.finditer(text):
            out.append(
                Violation(
                    rule="hygiene:todo-leftover",
                    message="Found TODO/FIXME.",
                    path=rel,
                    line=self._line_no(text, m.start()),
                    severity="info",
                )
            )

        # print() (skip test files)
        if not is_test_file:
            for m in self._RE_PRINT.finditer(text):
                out.append(
                    Violation(
                        rule="logging:no-print",
                        message="Replace print() with logging.",
                        path=rel,
                        line=self._line_no(text, m.start()),
                        severity="info",
                        hint="Use logging.getLogger(__name__).info(...).",
                    )
                )

        # long functions
        out.extend(
            self._long_function_violations(
                text=text, file_path=rel, max_len=self.long_function_limit
            )
        )

        return out

    # ---------- helpers ----------

    def _should_ignore(self, path: Path) -> bool:
        rel = self._relpath(path)
        for pat in self.ignore_globs:
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(path.name, pat):
                return True
        return False

    def _relpath(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except Exception:
            return str(path)

    @staticmethod
    def _line_no(text: str, pos: int) -> int:
        return text.count("\n", 0, pos) + 1

    def _long_function_violations(self, *, text: str, file_path: str, max_len: int) -> List[Violation]:
        """Naively detect function spans via next-def boundary."""
        out: List[Violation] = []
        lines = text.splitlines()
        starts: List[Tuple[int, int]] = []  # (line_idx, indent_spaces)

        for m in self._RE_DEF.finditer(text):
            line_idx = self._line_no(text, m.start()) - 1
            indent = len(m.group(1).replace("\t", "    "))
            starts.append((line_idx, indent))
        starts.append((len(lines), 0))  # sentinel

        for i in range(len(starts) - 1):
            start_idx, _ = starts[i]
            end_idx = starts[i + 1][0]
            length = max(0, end_idx - start_idx)
            if length > max_len:
                out.append(
                    Violation(
                        rule="design:long-function",
                        message=f"Function spans {length} lines (> {max_len}).",
                        path=file_path,
                        line=start_idx + 1,
                        severity="warning",
                        hint="Refactor into helpers.",
                        meta={"length": length, "limit": max_len},
                    )
                )
        return out


__all__ = ["Violation", "CodebasePhilosophyAuditor"]


if __name__ == "__main__":
    # Minimal CLI for ad-hoc use.
    root = Path(os.environ.get("AUDIT_ROOT", ".")).resolve()
    auditor = CodebasePhilosophyAuditor(root)
    violations = [v.to_dict() for v in auditor.scan()]
    print(json.dumps(violations, indent=2))
