# file: codebase_philosophy_auditor.py
from __future__ import annotations

"""
Analyze docstring coverage (module/class/function) across a Python codebase.

Outputs a concise plaintext summary by default or structured JSON with --json.
Designed to be mypy-friendly and resilient to syntax errors in scanned files.
"""

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union


AllowedDocNode = Union[ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]


def get_docstring_safe(node: ast.AST) -> Optional[str]:
    """
    Typed wrapper around ast.get_docstring.

    Only accepts Module/Class/Function/AsyncFunction nodes. Returns None when the
    node doesn't support docstrings or when no docstring is present.
    """
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return ast.get_docstring(node)
    return None


@dataclass(frozen=True)
class Undocumented:
    kind: str         # "module" | "class" | "function"
    name: str
    lineno: int


@dataclass(frozen=True)
class FileMetrics:
    path: Path
    module_name: str
    definitions: int
    documented_definitions: int
    classes: int
    functions: int
    has_module_docstring: bool
    undocumented: List[Undocumented]

    @property
    def documentation_ratio(self) -> float:
        if self.definitions == 0:
            return 1.0
        return self.documented_definitions / float(self.definitions)


def _iter_defs(tree: ast.AST) -> Iterator[AllowedDocNode]:
    if isinstance(tree, ast.Module):
        yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _module_name_for(path: Path, root: Path) -> str:
    """
    Derive a dotted module name relative to a root directory.
    Falls back to a filename-based name if path is outside root.
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
        return ".".join(Path(rel).with_suffix("").parts)
    except Exception:
        # Fallback: best-effort module-ish name from filename only
        return path.with_suffix("").name


def analyze_file(file_path: Path, *, root: Optional[Path] = None) -> FileMetrics:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(file_path), type_comments=True)
    module_name = _module_name_for(file_path, root or file_path.parent)

    has_module_doc = get_docstring_safe(tree) is not None

    defs = 0
    documented = 0
    cls_count = 0
    fn_count = 0
    missing: List[Undocumented] = []

    for node in _iter_defs(tree):
        if isinstance(node, ast.Module):
            # already accounted in has_module_doc
            continue

        defs += 1
        if isinstance(node, ast.ClassDef):
            cls_count += 1
            kind = "class"
            name = node.name
        else:
            fn_count += 1
            kind = "function"
            name = node.name  # type: ignore[attr-defined]

        if get_docstring_safe(node) is None:
            missing.append(Undocumented(kind=kind, name=name, lineno=getattr(node, "lineno", 0)))
        else:
            documented += 1

    return FileMetrics(
        path=file_path,
        module_name=module_name,
        definitions=defs,
        documented_definitions= documented,  # fixed: use the local counter
        classes=cls_count,
        functions=fn_count,
        has_module_docstring=has_module_doc,
        undocumented=missing,
    )


def analyze_path(path: Path) -> List[FileMetrics]:
    if path.is_file() and path.suffix == ".py":
        return [analyze_file(path, root=path.parent)]
    out: List[FileMetrics] = []
    for file in sorted(path.rglob("*.py")):
        out.append(analyze_file(file, root=path))
    return out


def _as_json(metrics: Sequence[FileMetrics]) -> str:
    payload = [
        {
            "path": str(m.path),
            "module": m.module_name,
            "definitions": m.definitions,
            "documented": m.documented_definitions,
            "ratio": m.documentation_ratio,
            "classes": m.classes,
            "functions": m.functions,
            "module_doc": m.has_module_docstring,
            "undocumented": [
                {"kind": u.kind, "name": u.name, "lineno": u.lineno} for u in m.undocumented
            ],
        }
        for m in metrics
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Codebase philosophy auditor (docstring coverage).")
    p.add_argument("path", nargs="?", default=".", help="File or directory to analyze (default: .)")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON metrics.")
    args = p.parse_args(argv)

    target = Path(args.path).resolve()
    try:
        result = analyze_path(target)
    except SyntaxError as e:
        print(f"SyntaxError in {getattr(e, 'filename', target)}:{getattr(e, 'lineno', '?')}: {e}", flush=True)
        return 2

    if args.json:
        print(_as_json(result))
    else:
        total_defs = sum(m.definitions for m in result)
        total_docd = sum(m.documented_definitions for m in result)
        ratio = (total_docd / total_defs) if total_defs else 1.0
        print(f"Analyzed {len(result)} file(s)")
        print(f"Docstring coverage: {total_docd}/{total_defs} = {ratio:.1%}")
        for m in result:
            # Quiet perfect files to keep output focused
            if not m.undocumented and m.has_module_docstring:
                continue
            print(f"\n{m.module_name} — doc={m.has_module_docstring}, defs={m.definitions}, covered={m.documented_definitions}")
            for u in m.undocumented:
                print(f"  {u.kind:8s} {u.lineno:5d}  {u.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
