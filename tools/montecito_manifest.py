“”“Utilities for generating file manifests for Montecito batch outputs.

This module provides a CLI that walks a directory, collects file sizes and
checksums (MD5 or SHA256), and writes a CSV manifest. It includes progress
reporting, error handling, and optional verification capabilities.

Features:
- Generate CSV manifests with filename, size, and checksum
- Progress indication for long-running operations
- Robust error handling for individual file failures
- Optional verification of existing manifests
- Support for both MD5 (legacy) and SHA256 checksums
- Duplicate file detection
“””

from **future** import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Iterable, Literal, NamedTuple, Optional

HashAlgorithm = Literal[“md5”, “sha256”]

class FileRecord(NamedTuple):
“”“Represents a single file’s metadata in the manifest.”””

```
path: Path
size: int
checksum: str
error: Optional[str] = None
```

class ManifestStats(NamedTuple):
“”“Statistics about manifest generation.”””

```
total_files: int
total_bytes: int
failed_files: int
duplicates: int
```

def compute_hash(path: Path, algorithm: HashAlgorithm = “md5”) -> str:
“”“Compute the hash of a file using the specified algorithm.

```
Args:
    path: Path to the file to hash.
    algorithm: Hash algorithm to use ('md5' or 'sha256').

Returns:
    Hexadecimal hash digest.

Raises:
    IOError: If the file cannot be read.
"""
hasher = hashlib.md5() if algorithm == "md5" else hashlib.sha256()

with path.open("rb") as file_obj:
    for chunk in iter(lambda: file_obj.read(65536), b""):  # 64KB chunks
        hasher.update(chunk)

return hasher.hexdigest()
```

def iter_files(
root: Path,
algorithm: HashAlgorithm = “md5”,
verbose: bool = False,
) -> Iterable[FileRecord]:
“”“Yield `FileRecord` for each file under `root`.

```
Args:
    root: Root directory to scan.
    algorithm: Hash algorithm to use.
    verbose: If True, print progress information.

Yields:
    FileRecord for each file found (including files that failed to process).
"""
all_files = sorted(root.rglob("*"))
file_count = sum(1 for p in all_files if p.is_file())

if verbose:
    print(f"Scanning {file_count:,} files...", file=sys.stderr)

processed = 0
for path in all_files:
    if not path.is_file():
        continue

    processed += 1

    # Progress indication
    if verbose and processed % 100 == 0:
        print(
            f"  Processed {processed:,}/{file_count:,} files...",
            file=sys.stderr,
            end="\r",
        )

    try:
        size = path.stat().st_size
        checksum = compute_hash(path, algorithm)
        yield FileRecord(path.relative_to(root), size, checksum)

    except PermissionError:
        error_msg = f"Permission denied"
        if verbose:
            print(
                f"\n⚠️  Skipping {path.relative_to(root)}: {error_msg}",
                file=sys.stderr,
            )
        yield FileRecord(path.relative_to(root), 0, "", error_msg)

    except OSError as e:
        error_msg = f"OS error: {e}"
        if verbose:
            print(
                f"\n⚠️  Skipping {path.relative_to(root)}: {error_msg}",
                file=sys.stderr,
            )
        yield FileRecord(path.relative_to(root), 0, "", error_msg)

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        if verbose:
            print(
                f"\n⚠️  Skipping {path.relative_to(root)}: {error_msg}",
                file=sys.stderr,
            )
        yield FileRecord(path.relative_to(root), 0, "", error_msg)

if verbose:
    print(f"\n✓ Completed scanning {file_count:,} files", file=sys.stderr)
```

def detect_duplicates(records: list[FileRecord]) -> dict[str, list[Path]]:
“”“Find files with identical checksums.

```
Args:
    records: List of file records to analyze.

Returns:
    Dictionary mapping checksums to lists of paths with that checksum.
    Only includes checksums that appear more than once.
"""
checksum_map: dict[str, list[Path]] = {}

for record in records:
    if record.error or not record.checksum:
        continue
    if record.checksum not in checksum_map:
        checksum_map[record.checksum] = []
    checksum_map[record.checksum].append(record.path)

# Filter to only duplicates
return {k: v for k, v in checksum_map.items() if len(v) > 1}
```

def write_manifest(
root: Path,
destination: Path,
algorithm: HashAlgorithm = “md5”,
verbose: bool = False,
check_duplicates: bool = False,
) -> ManifestStats:
“”“Write a CSV manifest for `root` to `destination`.

```
Args:
    root: Root directory to scan.
    destination: Path where manifest CSV will be written.
    algorithm: Hash algorithm to use.
    verbose: If True, print progress information.
    check_duplicates: If True, detect and report duplicate files.

Returns:
    ManifestStats with summary information.

Raises:
    IOError: If the destination cannot be written.
"""
# Collect all file records
records = [r for r in iter_files(root, algorithm, verbose)]

# Filter out failed files for the manifest
successful_records = [r for r in records if not r.error]
failed_count = len(records) - len(successful_records)

# Calculate statistics
total_bytes = sum(r.size for r in successful_records)

# Check for duplicates if requested
duplicates = detect_duplicates(successful_records) if check_duplicates else {}

# Write manifest
destination.parent.mkdir(parents=True, exist_ok=True)
hash_column = algorithm.upper()

with destination.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "bytes", hash_column.lower()])
    writer.writerows(
        (str(r.path), r.size, r.checksum) for r in successful_records
    )

# Report duplicates
if duplicates and verbose:
    print(f"\n⚠️  Found {len(duplicates)} duplicate file groups:", file=sys.stderr)
    for checksum, paths in list(duplicates.items())[:5]:  # Show first 5
        print(f"  {checksum}:", file=sys.stderr)
        for path in paths:
            print(f"    - {path}", file=sys.stderr)
    if len(duplicates) > 5:
        print(f"  ... and {len(duplicates) - 5} more", file=sys.stderr)

return ManifestStats(
    total_files=len(successful_records),
    total_bytes=total_bytes,
    failed_files=failed_count,
    duplicates=len(duplicates),
)
```

def verify_manifest(root: Path, manifest_path: Path, verbose: bool = False) -> bool:
“”“Verify that files match their manifest entries.

```
Args:
    root: Root directory containing the files.
    manifest_path: Path to the manifest CSV to verify.
    verbose: If True, print detailed progress.

Returns:
    True if all files match, False otherwise.
"""
if not manifest_path.exists():
    print(f"❌ Manifest not found: {manifest_path}", file=sys.stderr)
    return False

# Read manifest
with manifest_path.open("r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    manifest_entries = list(reader)

if verbose:
    print(f"Verifying {len(manifest_entries):,} files...", file=sys.stderr)

# Determine hash algorithm from header
first_entry = manifest_entries[0] if manifest_entries else {}
algorithm: HashAlgorithm = "sha256" if "sha256" in first_entry else "md5"

mismatches = []
missing = []

for i, entry in enumerate(manifest_entries, 1):
    if verbose and i % 100 == 0:
        print(
            f"  Verified {i:,}/{len(manifest_entries):,} files...",
            file=sys.stderr,
            end="\r",
        )

    file_path = root / entry["filename"]

    if not file_path.exists():
        missing.append(entry["filename"])
        continue

    try:
        actual_size = file_path.stat().st_size
        actual_hash = compute_hash(file_path, algorithm)

        expected_size = int(entry["bytes"])
        expected_hash = entry.get(algorithm, entry.get("md5", ""))

        if actual_size != expected_size or actual_hash != expected_hash:
            mismatches.append(
                {
                    "filename": entry["filename"],
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                    "expected_hash": expected_hash,
                    "actual_hash": actual_hash,
                }
            )

    except Exception as e:
        print(f"\n⚠️  Error verifying {entry['filename']}: {e}", file=sys.stderr)

if verbose:
    print(f"\n✓ Verification complete", file=sys.stderr)

# Report results
if missing:
    print(f"\n❌ Missing files: {len(missing)}", file=sys.stderr)
    for filename in missing[:10]:
        print(f"  - {filename}", file=sys.stderr)
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more", file=sys.stderr)

if mismatches:
    print(f"\n❌ Mismatched files: {len(mismatches)}", file=sys.stderr)
    for mismatch in mismatches[:10]:
        print(f"  - {mismatch['filename']}", file=sys.stderr)
        if mismatch["expected_size"] != mismatch["actual_size"]:
            print(
                f"    Size: {mismatch['expected_size']} → {mismatch['actual_size']}",
                file=sys.stderr,
            )
        if mismatch["expected_hash"] != mismatch["actual_hash"]:
            print(
                f"    Hash: {mismatch['expected_hash']} → {mismatch['actual_hash']}",
                file=sys.stderr,
            )
    if len(mismatches) > 10:
        print(f"  ... and {len(mismatches) - 10} more", file=sys.stderr)

success = not missing and not mismatches
if success:
    print(f"\n✅ All {len(manifest_entries):,} files verified successfully!")
else:
    print(
        f"\n❌ Verification failed: {len(missing)} missing, {len(mismatches)} mismatched"
    )

return success
```

def format_bytes(num_bytes: int) -> str:
“”“Format bytes as human-readable string.”””
for unit in [“B”, “KB”, “MB”, “GB”, “TB”]:
if num_bytes < 1024.0:
return f”{num_bytes:.1f} {unit}”
num_bytes /= 1024.0
return f”{num_bytes:.1f} PB”

def build_parser() -> argparse.ArgumentParser:
“”“Build argument parser for the CLI.”””
parser = argparse.ArgumentParser(
description=“Generate or verify CSV manifests (filename, size, checksum) for directories.”,
formatter_class=argparse.RawDescriptionHelpFormatter,
)

```
subparsers = parser.add_subparsers(dest="command", help="Command to execute")

# Generate command
generate_parser = subparsers.add_parser(
    "generate", help="Generate a new manifest"
)
generate_parser.add_argument(
    "root", type=Path, help="Root directory to scan for files"
)
generate_parser.add_argument(
    "destination",
    nargs="?",
    type=Path,
    default=Path("manifest.csv"),
    help="Where to write the manifest (default: ./manifest.csv)",
)
generate_parser.add_argument(
    "--algorithm",
    choices=["md5", "sha256"],
    default="md5",
    help="Hash algorithm to use (default: md5 for legacy compatibility)",
)
generate_parser.add_argument(
    "--check-duplicates",
    action="store_true",
    help="Detect and report duplicate files",
)
generate_parser.add_argument(
    "-v", "--verbose", action="store_true", help="Show progress information"
)

# Verify command
verify_parser = subparsers.add_parser(
    "verify", help="Verify files against an existing manifest"
)
verify_parser.add_argument(
    "root", type=Path, help="Root directory containing the files"
)
verify_parser.add_argument(
    "manifest", type=Path, help="Manifest file to verify against"
)
verify_parser.add_argument(
    "-v", "--verbose", action="store_true", help="Show detailed progress"
)

# For backward compatibility, allow direct invocation without subcommand
parser.add_argument(
    "root", type=Path, nargs="?", help="Root directory to scan (legacy mode)"
)
parser.add_argument(
    "destination",
    nargs="?",
    type=Path,
    help="Manifest destination (legacy mode)",
)

return parser
```

def main() -> None:
“”“Main CLI entry point.”””
parser = build_parser()
args = parser.parse_args()

```
# Handle legacy invocation (no subcommand)
if not args.command:
    if args.root is None:
        parser.print_help()
        raise SystemExit(1)

    # Legacy mode: generate manifest
    root = args.root.expanduser().resolve()
    destination = (
        (args.destination or Path("manifest.csv")).expanduser().resolve()
    )

    if not root.exists():
        raise SystemExit(f"❌ Root path '{root}' does not exist.")
    if not root.is_dir():
        raise SystemExit(f"❌ Root path '{root}' is not a directory.")

    stats = write_manifest(root, destination, "md5", verbose=True)

    print(f"\n{'='*60}")
    print(f"✅ Manifest written to: {destination}")
    print(f"{'='*60}")
    print(f"  Files:       {stats.total_files:,}")
    print(f"  Total size:  {format_bytes(stats.total_bytes)}")
    if stats.failed_files:
        print(f"  Failed:      {stats.failed_files:,}")
    print(f"{'='*60}")

    return

# Handle subcommands
if args.command == "generate":
    root = args.root.expanduser().resolve()
    destination = args.destination.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"❌ Root path '{root}' does not exist.")
    if not root.is_dir():
        raise SystemExit(f"❌ Root path '{root}' is not a directory.")

    stats = write_manifest(
        root,
        destination,
        args.algorithm,
        args.verbose,
        args.check_duplicates,
    )

    print(f"\n{'='*60}")
    print(f"✅ Manifest written to: {destination}")
    print(f"{'='*60}")
    print(f"  Algorithm:   {args.algorithm.upper()}")
    print(f"  Files:       {stats.total_files:,}")
    print(f"  Total size:  {format_bytes(stats.total_bytes)}")
    if stats.failed_files:
        print(f"  Failed:      {stats.failed_files:,}")
    if stats.duplicates:
        print(f"  Duplicates:  {stats.duplicates:,} groups")
    print(f"{'='*60}")

elif args.command == "verify":
    root = args.root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"❌ Root path '{root}' does not exist.")
    if not root.is_dir():
        raise SystemExit(f"❌ Root path '{root}' is not a directory.")

    success = verify_manifest(root, manifest_path, args.verbose)
    raise SystemExit(0 if success else 1)
```

if **name** == “**main**”:
main()