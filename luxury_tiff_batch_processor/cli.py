"""Command-line interface wiring for the luxury TIFF batch processor."""
from __future__ import annotations

import argparse
import dataclasses
import logging
import uuid
from pathlib import Path
from typing import Iterable, Optional

from .adjustments import AdjustmentSettings, LUXURY_PRESETS
from .pipeline import (
    _wrap_with_progress,
    collect_images,
    ensure_output_path,
    process_single_image,
)

LOGGER = logging.getLogger("luxury_tiff_batch_processor")


def default_output_folder(input_folder: Path) -> Path:
    """Return the default output folder for a given input directory."""

    if input_folder.name:
        return input_folder.parent / f"{input_folder.name}_lux"
    return input_folder / "luxury_output"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch enhance TIFF files for ultra-luxury marketing output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Folder that contains source TIFF files")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Folder where processed files will be written. Defaults to '<input>_lux' next to the input folder.",
    )
    parser.add_argument(
        "--preset",
        default="signature",
        choices=sorted(LUXURY_PRESETS.keys()),
        help="Adjustment preset that provides a starting point",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process folders recursively and mirror the directory tree in the output",
    )
    parser.add_argument(
        "--suffix",
        default="_lux",
        help="Filename suffix appended before the extension for processed files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the destination",
    )
    parser.add_argument(
        "--compression",
        default="tiff_lzw",
        help="TIFF compression to use when saving (as understood by Pillow)",
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=None,
        help="Optionally resize the longest image edge to this many pixels while preserving aspect ratio",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview the work without writing any files")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting (useful for minimal or non-interactive environments)",
    )

    # Fine control overrides.
    parser.add_argument("--exposure", type=float, default=None, help="Exposure adjustment in stops")
    parser.add_argument(
        "--white-balance-temp",
        type=float,
        default=None,
        dest="white_balance_temp",
        help="Target color temperature in Kelvin",
    )
    parser.add_argument(
        "--white-balance-tint",
        type=float,
        default=None,
        dest="white_balance_tint",
        help="Green-magenta tint compensation (positive skews magenta)",
    )
    parser.add_argument("--shadow-lift", type=float, default=None, help="Shadow recovery strength (0-1)")
    parser.add_argument(
        "--highlight-recovery",
        type=float,
        default=None,
        help="Highlight compression strength (0-1)",
    )
    parser.add_argument(
        "--midtone-contrast", type=float, default=None, dest="midtone_contrast", help="Midtone contrast strength"
    )
    parser.add_argument("--vibrance", type=float, default=None, help="Vibrance strength (0-1)")
    parser.add_argument("--saturation", type=float, default=None, help="Additional saturation multiplier delta")
    parser.add_argument("--clarity", type=float, default=None, help="Local contrast boost strength (0-1)")
    parser.add_argument(
        "--chroma-denoise",
        type=float,
        default=None,
        dest="chroma_denoise",
        help="Chrominance denoising amount (0-1)",
    )
    parser.add_argument(
        "--luxury-glow", type=float, default=None, dest="glow", help="Diffusion glow strength (0-1)"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output is None:
        args.output = default_output_folder(args.input)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    return args


def build_adjustments(args: argparse.Namespace) -> AdjustmentSettings:
    base = dataclasses.replace(LUXURY_PRESETS[args.preset])
    for field in dataclasses.fields(base):
        value = getattr(args, field.name, None)
        if value is not None:
            setattr(base, field.name, value)
            base._validate()
    LOGGER.debug("Using adjustments: %s", base)
    return base


def _ensure_non_overlapping(input_root: Path, output_root: Path) -> None:
    def _contains(parent: Path, child: Path) -> bool:
        try:
            child.relative_to(parent)
        except ValueError:
            return False
        return True

    if input_root == output_root:
        raise SystemExit("Output folder must be different from the input folder to avoid self-overwrites.")
    if _contains(input_root, output_root):
        raise SystemExit(
            "Output folder cannot be located inside the input folder; choose a sibling or separate directory."
        )
    if _contains(output_root, input_root):
        raise SystemExit(
            "Input folder cannot be located inside the output folder; choose non-overlapping directories."
        )


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the batch processor with the provided arguments."""

    run_id = uuid.uuid4().hex
    adjustments = build_adjustments(args)
    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    if not input_root.is_dir():
        raise SystemExit(f"Input folder '{input_root}' does not exist or is not a directory")

    _ensure_non_overlapping(input_root, output_root)

    LOGGER.info("Starting batch run %s for %s", run_id, input_root)
    images = sorted(collect_images(input_root, args.recursive))
    if not images:
        LOGGER.warning("No TIFF images found in %s (run %s)", input_root, run_id)
        return 0

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Found %s image(s) to process", len(images))
    processed = 0

    progress_iterable = _wrap_with_progress(
        images,
        total=len(images),
        description="Processing images",
        enabled=not getattr(args, "no_progress", False),
    )

    for image_path in progress_iterable:
        destination = ensure_output_path(
            input_root,
            output_root,
            image_path,
            args.suffix,
            args.recursive,
            create=not args.dry_run,
        )
        if destination.exists() and not args.overwrite and not args.dry_run:
            LOGGER.warning("Skipping %s (exists, use --overwrite to replace)", destination)
            continue
        if args.dry_run:
            LOGGER.info("Dry run: would process %s -> %s", image_path, destination)
        process_single_image(
            image_path,
            destination,
            adjustments,
            compression=args.compression,
            resize_long_edge=args.resize_long_edge,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            processed += 1

    LOGGER.info("Finished batch run %s; processed %s image(s)", run_id, processed)
    return processed


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


__all__ = [
    "build_adjustments",
    "default_output_folder",
    "main",
    "parse_args",
    "run_pipeline",
]
