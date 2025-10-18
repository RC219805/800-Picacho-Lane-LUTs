# path: tools/montecito_manifest.py
"""Utilities for generating file manifests for Montecito batch outputs.

This module provides a CLI that scans a directory tree, collects file sizes and
MD5 checksums (to match historical `md5 -q`), and writes a CSV manifest:
`filename,bytes,md5`. Defaults are drop-in compatible with prior shell scripts.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple


# --- Core helpers ------------------------------------------------------------

def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Fast, incremental MD5; chunk_size tuned for spinning + SSD. """
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileEntry:
    rel_posix: str  # posix-style relative path ("a/b/c.ext")
    abs_path: Path
    size: int


def _match_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _collect_files(
    root: Path,
    *,
    include: Sequence[str],
    exclude: Sequence[str],
    follow_symlinks: bool,
) -> List[FileEntry]:
    """Return a deterministic, filtered list of files under root."""
    entries: List[FileEntry] = []
    # os.walk is faster than rglob for large trees; we normalize to Paths.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        # Deterministic traversal (casefold: stable on Windows/macOS)
        dirnames.sort(key=str.casefold)
        filenames.sort(key=str.casefold)

        # Optionally prune excluded directories early
        if exclude:
            dirnames[:] = [
                d for d in dirnames
                if not _match_any(os.path.join(os.path.relpath(os.path.join(dirpath, d), root), ""), exclude)
            ]

        for name in filenames:
            abs_path = Path(dirpath) / name
            try:
                if not abs_path.is_file():
                    continue
            except OSError:
                continue

            rel = abs_path.relative_to(root)
            rel_str = rel.as_posix()

            # Include/exclude filtering on the relative POSIX path
            if include and not _match_any(rel_str, include):
                continue
            if exclude and _match_any(rel_str, exclude):
                continue

            try:
                size = abs_path.stat().st_size
            except OSError:
                continue

            entries.append(FileEntry(rel_posix=rel_str, abs_path=abs_path, size=size))

    # Final deterministic order
    entries.sort(key=lambda e: e.rel_posix.casefold())
    return entries


# --- Public API --------------------------------------------------------------

def iter_files(root: Path) -> Iterable[Tuple[Path, int, str]]:
    """Yield (relative_path, size_bytes, md5_hash) for each file under root.

    Kept for backward compatibility with earlier scripts. Uses linear hashing.
    """
    for entry in _collect_files(root, include=(), exclude=(), follow_symlinks=False):
        yield Path(entry.rel_posix), entry.size, _md5(entry.abs_path)


def write_manifest(
    root: Path,
    destination: Path,
    *,
    workers: int = 1,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    follow_symlinks: bool = False,
    skip_errors: bool = False,
) -> None:
    """Write a CSV manifest for `root` to `destination`.

    CSV headers: `filename,bytes,md5`. Paths are POSIX relative to `root`.
    `workers>1` enables threaded hashing (I/O bound speedup on many files).
    """
    files = _collect_files(
        root, include=include, exclude=exclude, follow_symlinks=follow_symlinks
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "bytes", "md5"])

        if workers <= 1 or len(files) < 2:
            # Linear path: simplest and most predictable
            for e in files:
                try:
                    md5 = _md5(e.abs_path)
                    writer.writerow([e.rel_posix, e.size, md5])
                except Exception as exc:
                    if skip_errors:
                        print(f"[skip] {e.rel_posix}: {exc}", file=sys.stderr)
                        continue
                    raise
            return

        # Threaded hashing preserving input order
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_md5, e.abs_path): idx
                for idx, e in enumerate(files)
            }
            results: List[str | None] = [None] * len(files)

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    if skip_errors:
                        print(f"[skip] {files[idx].rel_posix}: {exc}", file=sys.stderr)
                        results[idx] = ""  # placeholder; will be skipped
                    else:
                        raise

            for e, md5 in zip(files, results):
                if md5 is None or md5 == "":
                    if skip_errors:
                        continue
                    raise RuntimeError(f"Missing hash for {e.rel_posix}")
                writer.writerow([e.rel_posix, e.size, md5])


# --- CLI ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a CSV manifest (filename, bytes, md5) for an output directory."
    )
    parser.add_argument("root", type=Path, help="Root directory to scan.")
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=Path("manifest.csv"),
        help="Where to write the manifest (defaults to ./manifest.csv).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Threaded hashing workers (I/O bound speedup). 1 = disabled.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob to include (relative POSIX paths). May be given multiple times. Example: 'processed_images/**/*.tif*'",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob to exclude (relative POSIX paths). May be given multiple times. Example: '**/.DS_Store'",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow directory symlinks during traversal.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip unreadable files with a stderr note instead of failing.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root: Path = args.root.expanduser().resolve()
    destination: Path = args.destination.expanduser().resolve()

    if not root.exists():
        raise SystemExit(
            f"Root path '{root}' does not exist. Please check the path."
        )
    if not root.is_dir():
        raise SystemExit(
            f"Root path '{root}' exists but is not a directory."
        )
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    write_manifest(
        root,
        destination,
        workers=int(args.workers or 1),
        include=tuple(args.include or ()),
        exclude=tuple(args.exclude or ()),
        follow_symlinks=bool(args.follow_symlinks),
        skip_errors=bool(args.skip_errors),
    )


if __name__ == "__main__":
    main()