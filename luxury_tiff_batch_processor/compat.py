# path: luxury_tiff_batch_processor/compat.py
"""Compatibility layer for legacy argparse-style API.

This module provides backward compatibility for code that was written against
an older argparse-based interface. The new implementation uses Typer for CLI
operations, but these functions bridge the gap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .adjustments import LUXURY_PRESETS
from .pipeline import collect_images, process_single_image


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Legacy argparse-style CLI parser for backward compatibility.

    Parameters
    ----------
    argv : List[str]
        Command-line arguments (without program name)

    Returns
    -------
    argparse.Namespace
        Parsed arguments compatible with run_pipeline
    """
    parser = argparse.ArgumentParser(description="Luxury TIFF Batch Processor (compat)")

    parser.add_argument("input", type=Path, help="Input directory")
    parser.add_argument("output", nargs="?", type=Path, help="Output directory")
    parser.add_argument("--preset", type=str, default=None, help="Preset name")
    parser.add_argument("--recursive", action="store_true", help="Process recursively")
    parser.add_argument("--suffix", type=str, default="_lux", help="Output suffix")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--compression", type=str, default="tiff_lzw", help="TIFF compression")
    parser.add_argument("--resize-long-edge", type=int, default=None, help="Resize long edge")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")

    # Adjustment parameters
    parser.add_argument("--exposure", type=float, default=None)
    parser.add_argument("--white-balance-temp", type=float, default=None)
    parser.add_argument("--white-balance-tint", type=float, default=None)
    parser.add_argument("--shadow-lift", type=float, default=None)
    parser.add_argument("--highlight-recovery", type=float, default=None)
    parser.add_argument("--midtone-contrast", type=float, default=None)
    parser.add_argument("--vibrance", type=float, default=None)
    parser.add_argument("--saturation", type=float, default=None)
    parser.add_argument("--clarity", type=float, default=None)
    parser.add_argument("--chroma-denoise", type=float, default=None)
    parser.add_argument("--luxury-glow", type=float, default=None, dest="glow")

    args = parser.parse_args(argv)

    # Set output default if not provided
    if args.output is None:
        args.output = Path(str(args.input) + "_processed")

    return args


def run_pipeline(args: argparse.Namespace) -> int:
    """Legacy pipeline runner for backward compatibility.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from parse_args

    Returns
    -------
    int
        Number of successfully processed images
    """
    from .adjustments import AdjustmentSettings

    # Get preset settings if specified
    if args.preset:
        if args.preset not in LUXURY_PRESETS:
            raise ValueError(f"Unknown preset: {args.preset}")
        settings = LUXURY_PRESETS[args.preset]
    else:
        # Use default or build from args
        settings = AdjustmentSettings()

    # Apply any overrides from args
    overrides = {}
    for attr in [
        "exposure", "white_balance_temp", "white_balance_tint",
        "shadow_lift", "highlight_recovery", "midtone_contrast",
        "vibrance", "saturation", "clarity", "chroma_denoise", "glow"
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            overrides[attr] = value

    if overrides:
        # Create new settings with overrides
        from dataclasses import replace
        settings = replace(settings, **overrides)

    # If dry run, just return 0
    if getattr(args, "dry_run", False):
        return 0

    # Collect input images
    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Collect images (recursive mode based on args)
    images = collect_images(input_dir, recursive=getattr(args, "recursive", False))

    if not images:
        return 0

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    processed_count = 0
    suffix = getattr(args, "suffix", "_lux")
    overwrite = getattr(args, "overwrite", False)

    for img_path in images:
        # Build output path
        rel_path = img_path.relative_to(input_dir)
        out_name = f"{rel_path.stem}{suffix}{rel_path.suffix}"
        out_path = output_dir / rel_path.parent / out_name

        # Skip if exists and not overwrite
        if out_path.exists() and not overwrite:
            continue

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Process the image
            process_single_image(
                img_path,
                out_path,
                settings,
                resize_target=getattr(args, "resize_long_edge", None),
            )
            processed_count += 1
        except Exception as e:
            # Log error but continue processing
            print(f"Error processing {img_path}: {e}", file=sys.stderr)
            continue

    return processed_count


__all__ = ["parse_args", "run_pipeline"]
