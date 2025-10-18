# file: tools/ndarray_typing_migrator.py
"""
Migrate 'np.ndarray' / 'numpy.ndarray' annotations to 'NDArray' with TYPE_CHECKING stubs.

- Rewrites ONLY in:
  1) String annotations (e.g., "np.ndarray", 'numpy.ndarray', possibly with [T]).
  2) Type comments (e.g.,  # type: np.ndarray[...]).

- Leaves runtime references like isinstance(x, np.ndarray) untouched.

Usage:
  Dry run (CI-safe):   python tools/ndarray_typing_migrator.py --check
  Apply changes:       python tools/ndarray_typing_migrator.py --fix
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

# --------- discover files ----------------------------------------------------

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".venv", "venv", "env", "__pycache__", "build", "dist", "site-packages",
    ".eggs", ".idea", ".vscode", ".github",
}

def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        parts = set(p.parts)
        if parts & SKIP_DIRS:
            continue
        yield p

# --------- regexes -----------------------------------------------------------

# String annotation forms: preserve quotes and any [ ... ] generic suffix.
# Examples matched: "np.ndarray", 'numpy.ndarray', "np.ndarray[Any]", etc.
STR_ANN_RE = re.compile(
    r"""
    (?P<q>['"])              # opening quote
    (?P<prefix>np|numpy)     # np or numpy
    \.ndarray
    (?P<gen>\[[^\]]*\])?     # optional generic [ ... ]
    (?P=q)                   # closing quote (same as opening)
    """,
    re.VERBOSE,
)

# Type comment forms:  # type: np.ndarray[...]   or   # type: "np.ndarray[...]"
TYPE_COMMENT_RE = re.compile(
    r"""
    (?P<head>\#\s*type:\s*)          # leading comment header
    (?P<q>['"])?                     # optional quote
    (?P<prefix>np|numpy)\.ndarray    # np.ndarray / numpy.ndarray
    (?P<gen>\[[^\]]*\])?             # optional generic
    (?P=q)?                          # optional matching quote
    """,
    re.VERBOSE,
)

# Detect existing TYPE_CHECKING import / stub
TYPE_CHECKING_IMPORT_RE = re.compile(r"^\s*from\s+typing\s+import\s+.*\bTYPE_CHECKING\b", re.MULTILINE)
TYPING_IMPORT_LINE_RE = re.compile(r"^\s*from\s+typing\s+import\s+([^\n]+)$", re.MULTILINE)
TYPE_CHECKING_BLOCK_RE = re.compile(r"^\s*if\s+TYPE_CHECKING\s*:\s*(?:#.*)?$", re.MULTILINE)
NDARRAY_IMPORT_LINE = "from numpy.typing import NDArray  # noqa: F401\n"
TYPE_CHECKING_BLOCK = (
    "if TYPE_CHECKING:  # pragma: no cover\n"
    f"    {NDARRAY_IMPORT_LINE}"
)

# Shebang / encoding / module docstring / future import helpers
SHEBANG_RE = re.compile(r"^#![^\n]*\n")
ENCODING_RE = re.compile(r"^#.*coding[:=]\s*([-\w.]+)", re.MULTILINE)
FUTURE_IMPORT_RE = re.compile(r"^\s*from\s+__future__\s+import\s+.*$", re.MULTILINE)
DOCSTRING_START_RE = re.compile(r"^\s*(?:['\"]).*?(?:['\"])")  # heuristic

@dataclass
class FileEdit:
    path: Path
    changed: bool
    did_insert_type_checking: bool
    did_insert_block: bool
    replacements: int

# --------- core rewrite ------------------------------------------------------

def rewrite_text(src: str) -> Tuple[str, int]:
    """
    Returns (new_text, num_replacements). Only updates string annotations and type comments.
    """
    def _str_repl(m: re.Match) -> str:
        q = m.group("q")
        gen = m.group("gen") or ""
        return f"{q}NDArray{gen}{q}"

    def _cmt_repl(m: re.Match) -> str:
        head = m.group("head")
        q = m.group("q") or ""
        gen = m.group("gen") or ""
        if q:
            return f"{head}{q}NDArray{gen}{q}"
        return f"{head}NDArray{gen}"

    new = STR_ANN_RE.sub(_str_repl, src)
    n1 = len(STR_ANN_RE.findall(src))

    newer = TYPE_COMMENT_RE.sub(_cmt_repl, new)
    n2 = len(TYPE_COMMENT_RE.findall(new))

    return newer, (n1 + n2)

def ensure_type_checking_stub(text: str, changed_replacements: int) -> Tuple[str, bool, bool]:
    """
    If replacements occurred and no TYPE_CHECKING/NDArray import exists, inject them near imports.
    Returns (new_text, added_import, added_block).
    """
    added_import = False
    added_block = False

    if changed_replacements == 0:
        return text, added_import, added_block

    t = text

    # Ensure "from typing import TYPE_CHECKING"
    if not TYPE_CHECKING_IMPORT_RE.search(t):
        # Try to extend an existing 'from typing import ...' line
        m = TYPING_IMPORT_LINE_RE.search(t)
        if m:
            whole = m.group(0)
            names = [x.strip() for x in m.group(1).split(",")]
            if "TYPE_CHECKING" not in names:
                new_line = whole.rstrip() + ", TYPE_CHECKING\n"
                t = t.replace(whole, new_line, 1)
                added_import = True
        else:
            # Insert a fresh import after shebang/encoding/docstring/future imports
            insert_idx = 0
            # skip shebang
            m_she = SHEBANG_RE.match(t)
            if m_she:
                insert_idx = m_she.end()
            # skip encoding cookie line if right at top
            m_enc = ENCODING_RE.match(t)
            if m_enc and m_enc.start() == 0:
                insert_idx = ENCODING_RE.search(t).end()  # type: ignore
            # skip leading blank lines/comments
            while insert_idx < len(t) and t[insert_idx] in "\r\n":
                insert_idx += 1
            # if module docstring present at very top, skip it
            if DOCSTRING_START_RE.match(t[insert_idx:]):
                quote = t[insert_idx]
                # naive find matching triple or single
                if t[insert_idx:insert_idx+3] == quote*3:
                    end = t.find(quote*3, insert_idx+3)
                    if end != -1:
                        insert_idx = end + 3
                else:
                    end = t.find(quote, insert_idx+1)
                    if end != -1:
                        insert_idx = end + 1
                # trailing newlines after docstring
                while insert_idx < len(t) and t[insert_idx] in "\r\n ":
                    insert_idx += 1
            # move past any __future__ imports block
            futs = list(FUTURE_IMPORT_RE.finditer(t))
            if futs:
                insert_idx = max(insert_idx, futs[-1].end())
            # inject the import
            t = t[:insert_idx] + "\nfrom typing import TYPE_CHECKING\n" + t[insert_idx:]
            added_import = True

    # Ensure TYPE_CHECKING block with NDArray import
    if not TYPE_CHECKING_BLOCK_RE.search(t) or "numpy.typing import NDArray" not in t:
        # place the block after the TYPE_CHECKING import (best-effort)
        pos = t.find("from typing import TYPE_CHECKING")
        if pos == -1:
            # fallback: place after first import block
            pos = max(t.find("import "), t.find("from "))
            if pos == -1:
                pos = 0
        # Insert two newlines before the block if preceding token not newline
        insert_at = t.find("\n", pos)
        if insert_at == -1:
            insert_at = len(t)
        insertion = "\n" + TYPE_CHECKING_BLOCK + "\n"
        if "numpy.typing import NDArray" not in t:
            t = t[:insert_at+1] + insertion + t[insert_at+1:]
            added_block = True

    return t, added_import, added_block

# --------- driver ------------------------------------------------------------

@dataclass
class Summary:
    edited: List[FileEdit]

    def print(self) -> None:
        if not self.edited:
            print("No files needed migration.")
            return
        total_rep = sum(e.replacements for e in self.edited)
        print(f"Migrated {len(self.edited)} files, {total_rep} replacements.")
        for e in self.edited:
            flags = []
            if e.did_insert_type_checking: flags.append("add TYPE_CHECKING")
            if e.did_insert_block: flags.append("add NDArray stub")
            flag_s = f" ({', '.join(flags)})" if flags else ""
            print(f" - {e.path}{flag_s}: {e.replacements} repl")

def process(root: Path, fix: bool) -> Summary:
    edits: List[FileEdit] = []
    for f in iter_py_files(root):
        src = f.read_text(encoding="utf-8")
        new, nrepl = rewrite_text(src)
        if nrepl == 0:
            continue
        new2, add_imp, add_blk = ensure_type_checking_stub(new, nrepl)
        changed = (new2 != src)
        edits.append(FileEdit(f, changed, add_imp, add_blk, nrepl))
        if fix and changed:
            f.write_text(new2, encoding="utf-8")
    return Summary(edits)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="Apply changes in-place.")
    ap.add_argument("--check", action="store_true", help="Dry-run and print summary (default).")
    ap.add_argument("--root", default=".", help="Repository root (default: .)")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    # default to check-mode if neither flag given
    fix = bool(args.fix)
    if not args.fix and not args.check:
        print("No mode selected; defaulting to --check")
    summary = process(root, fix=fix)
    summary.print()

    # In CI, you could fail if changes are needed and not applied.
    if not fix and summary.edited:
        # Exit 0 by default; flip to 1 if you want to enforce running --fix locally.
        return 0
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
