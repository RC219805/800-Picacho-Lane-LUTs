# path: luxury_tiff_batch_processor/pipeline.py
"""Core processing helpers shared between the CLI and integrations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Literal, Union

import numpy as np

from .adjustments import AdjustmentSettings, apply_adjustments, batch_apply_adjustments
from .io_utils import (
    ProcessingContext,
    float_to_dtype_array,
    image_to_float,
    save_image,
)
from .profiles import (
    DEFAULT_PROFILE_NAME,
    PROCESSING_PROFILES,
    ProcessingProfile,
)


def _ensure_profile(profile: ProcessingProfile | None) -> ProcessingProfile:
    return PROCESSING_PROFILES[DEFAULT_PROFILE_NAME] if profile is None else profile


try:  # optional progress bar
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None

LOGGER = logging.getLogger("luxury_tiff_batch_processor")
WORKER_LOGGER = LOGGER.getChild("worker")


def _tqdm_progress(
    iterable: Iterable[object], *, total: Optional[int], description: Optional[str]
) -> Iterable[object]:
    """Wrap iterable with tqdm if available."""
    if _tqdm is None:  # pragma: no cover
        return iterable
    return _tqdm(iterable, total=total, desc=description, unit="image")


_PROGRESS_WRAPPER = _tqdm_progress if _tqdm is not None else None


def _wrap_with_progress(
    iterable: Iterable[Path],
    *,
    total: Optional[int],
    description: str,
    enabled: bool,
) -> Iterable[Path]:
    """Return iterable wrapped with a progress helper when available."""
    if not enabled:
        return iterable
    helper = _PROGRESS_WRAPPER
    if helper is None:
        LOGGER.debug("Progress helper not available; install tqdm for progress reporting.")
        return iterable
    try:
        return helper(iterable, total=total, description=description)
    except Exception:  # pragma: no cover
        LOGGER.exception("Progress helper failed; continuing without progress display.")
        return iterable


def collect_images(folder: Path, recursive: bool) -> Iterator[Path]:
    patterns: List[str] = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    if recursive:
        for pattern in patterns:
            yield from folder.rglob(pattern)
    else:
        for pattern in patterns:
            yield from folder.glob(pattern)


def ensure_output_path(
    input_root: Path,
    output_root: Path,
    source: Path,
    suffix: str,
    recursive: bool,
    *,
    create: bool = True,
) -> Path:
    relative = source.relative_to(input_root) if recursive else Path(source.name)
    destination = output_root / relative
    if create:
        destination.parent.mkdir(parents=True, exist_ok=True)
    new_name = destination.stem + suffix + destination.suffix
    return destination.with_name(new_name)


def resize_bilinear(arr: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    height, width = arr.shape[:2]
    if width == new_width and height == new_height:
        return arr
    x = np.linspace(0, width - 1, new_width, dtype=np.float32)
    y = np.linspace(0, height - 1, new_height, dtype=np.float32)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x_weight = (x - x0).astype(np.float32).reshape(1, -1, 1)
    y_weight = (y - y0).astype(np.float32).reshape(-1, 1, 1)

    Ia = arr[np.ix_(y0, x0)]
    Ib = arr[np.ix_(y0, x1)]
    Ic = arr[np.ix_(y1, x0)]
    Id = arr[np.ix_(y1, x1)]

    top = Ia * (1.0 - x_weight) + Ib * x_weight
    bottom = Ic * (1.0 - x_weight) + Id * x_weight
    return (top * (1.0 - y_weight) + bottom * y_weight).astype(np.float32)


def resize_array(arr: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    if arr.ndim == 2:
        expanded = arr[:, :, None]
        resized = resize_bilinear(expanded, new_width, new_height)
        return resized[:, :, 0]
    if arr.ndim == 3:
        return resize_bilinear(arr, new_width, new_height)
    raise ValueError("Unsupported array shape for resizing")


def resize_long_edge_array(arr: np.ndarray, target: int) -> np.ndarray:
    height, width = arr.shape[:2]
    long_edge = max(width, height)
    if long_edge <= target:
        return arr
    scale = target / float(long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    LOGGER.debug("Resizing from %sx%s to %s", width, height, (new_width, new_height))
    return resize_array(arr, new_width, new_height)


def _coerce_resize_target(
    resize_long_edge: Optional[int], resize_target: Optional[int]
) -> Optional[int]:
    """Normalize legacy `resize_target` usage."""
    if resize_long_edge is None:
        return resize_target
    if resize_target is None or resize_target == resize_long_edge:
        return resize_long_edge
    raise ValueError("Conflicting resize targets provided; choose one value")


def _process_image_worker(
    source: Path,
    destination: Path,
    adjustments: AdjustmentSettings,
    *,
    compression: str,
    resize_long_edge: Optional[int] = None,
    resize_target: Optional[int] = None,
    dry_run: bool = False,
    profile: ProcessingProfile,
) -> bool:
    """Process a single image; safe for ProcessPoolExecutor.

    Why: avoids `PIL.Image` handling here; `image_to_float` captures metadata/ICC.
    """
    WORKER_LOGGER.info("Processing %s -> %s", source, destination)
    if destination.exists() and not dry_run and not destination.is_file():
        if destination.is_dir():
            path_type = "directory"
        elif destination.is_symlink():
            path_type = "symlink"
        else:
            path_type = "non-file"
        raise ValueError(f"Destination path exists but is a {path_type}: {destination}")

    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)

    # Load + normalize + capture metadata/ICC
    float_result = image_to_float(source, return_format="object")
    arr = float_result.array
    dtype_in = float_result.dtype
    alpha = float_result.alpha
    base_channels = float_result.base_channels
    float_norm = float_result.float_normalisation

    effective_profile = _ensure_profile(profile)

    # Color pipeline
    adjusted = apply_adjustments(arr, adjustments, profile=effective_profile)

    # Optional resize (RGB + alpha)
    target = _coerce_resize_target(resize_long_edge, resize_target)
    if target is not None:
        adjusted = resize_long_edge_array(adjusted, target)
        if alpha is not None:
            alpha = resize_long_edge_array(alpha, target)

    # Choose output dtype via profile; convert floats to that dtype (preserve alpha)
    target_dtype = effective_profile.target_dtype(dtype_in)
    arr_int = float_to_dtype_array(
        adjusted,
        target_dtype,
        alpha,
        base_channels,
        float_normalisation=float_norm,
    )

    if dry_run:
        WORKER_LOGGER.info("Dry run enabled, skipping save for %s", destination)
        return False

    # Atomic write; preserve metadata/ICC
    with ProcessingContext(destination) as staged_path:
        save_image(
            staged_path,
            arr_int,
            target_dtype,
            float_result.metadata,
            float_result.icc_profile,
            compression,
        )
    return True


def process_single_image(
    source: Path,
    destination: Path,
    adjustments: AdjustmentSettings,
    *,
    compression: str,
    resize_long_edge: Optional[int] = None,
    resize_target: Optional[int] = None,
    dry_run: bool = False,
    profile: ProcessingProfile | None = None,
) -> None:
    """Public wrapper around `_process_image_worker`."""
    _process_image_worker(
        source,
        destination,
        adjustments,
        compression=compression,
        resize_long_edge=resize_long_edge,
        resize_target=resize_target,
        dry_run=dry_run,
        profile=_ensure_profile(profile),
    )


# Backwards compatibility shim for older integrations.
process_image = process_single_image


# ----------------------- NEW: batch processing helper ------------------------


def process_images_batch(
    images: Sequence[Path],
    input_root: Path,
    output_root: Path,
    adjustments: Union[AdjustmentSettings, Sequence[AdjustmentSettings]],
    *,
    profile: ProcessingProfile | None = None,
    compression: str = "tiff_lzw",
    suffix: str = "_lux",
    recursive: bool = True,
    method: Literal["auto", "vectorized", "multiprocessing"] = "auto",
    workers: Optional[int] = None,
    resize_long_edge: Optional[int] = None,
    resize_target: Optional[int] = None,
    dry_run: bool = False,
    show_progress: bool = True,
    overwrite: bool = False,
) -> int:
    """Batch-process images with vectorized groups and save to mirrored paths.

    Uses `batch_apply_adjustments` per (H, W) bucket for speed; preserves metadata/ICC
    and converts dtype via `ProcessingProfile`.
    """
    eff_profile = _ensure_profile(profile)
    target = _coerce_resize_target(resize_long_edge, resize_target)

    # Build destinations and filter worklist.
    to_process: List[int] = []
    destinations: List[Path] = []
    for idx, src in enumerate(images):
        dst = ensure_output_path(
            input_root, output_root, src, suffix, recursive, create=not dry_run
        )
        destinations.append(dst)
        if dst.exists() and not overwrite and not dry_run:
            LOGGER.warning("Skipping %s (exists, use overwrite to replace)", dst)
            continue
        to_process.append(idx)

    if not to_process:
        return 0

    # Progress over saves, independent of grouping.
    total = len(to_process)
    progress_iter = _wrap_with_progress(
        (images[i] for i in to_process),
        total=total,
        description="Batch processing",
        enabled=show_progress,
    )
    progress_iter = iter(progress_iter)

    def _tick() -> None:
        try:
            next(progress_iter)
        except StopIteration:
            pass

    # Load + bucket by shape while keeping per-file metadata.
    from collections import defaultdict

    BucketItem = Tuple[int, np.ndarray, np.dtype, Optional[np.ndarray], int, object, Optional[bytes]]
    buckets: dict[Tuple[int, int], List[BucketItem]] = defaultdict(list)

    for idx in to_process:
        src = images[idx]
        res = image_to_float(src, return_format="object")
        arr = res.array  # (H,W,3) float32 [0,1]
        h, w = int(arr.shape[0]), int(arr.shape[1])
        # Pack dtype/alpha/base_channels/float_norm/icc for later per-file conversion.
        buckets[(h, w)].append(
            (idx, arr, res.dtype, res.alpha, res.base_channels, res.float_normalisation, res.icc_profile)
        )

    processed = 0

    # Process each bucket using batch_apply_adjustments.
    for (h, w), items in buckets.items():
        inds = [it[0] for it in items]
        arrays = [it[1] for it in items]
        dtypes_in = [it[2] for it in items]
        alphas = [it[3] for it in items]
        base_channels_list = [it[4] for it in items]
        norms = [it[5] for it in items]
        iccs = [it[6] for it in items]

        batch = np.stack(arrays, axis=0).astype(np.float32, copy=False)

        # Determine settings for this bucket.
        if isinstance(adjustments, AdjustmentSettings):
            adj_param: Union[AdjustmentSettings, Sequence[AdjustmentSettings]] = adjustments
        else:
            adj_param = [adjustments[i] for i in inds]

        try:
            out_batch = batch_apply_adjustments(
                batch, adj_param, method=method, workers=workers
            )
        except Exception as e:
            LOGGER.warning("[vectorized->serial fallback] (%s,%s): %s", h, w, e)
            if isinstance(adj_param, AdjustmentSettings):
                out_batch = np.stack(
                    [apply_adjustments(batch[i], adj_param, profile=eff_profile) for i in range(batch.shape[0])],
                    axis=0,
                )
            else:
                out_batch = np.stack(
                    [apply_adjustments(batch[i], adj_param[i], profile=eff_profile) for i in range(batch.shape[0])],
                    axis=0,
                )

        # Save per file (resize/alpha/metadata/dtype are per-file concerns).
        for slot, idx in enumerate(inds):
            out = out_batch[slot]
            alpha = alphas[slot]
            if target is not None:
                out = resize_long_edge_array(out, target)
                if alpha is not None:
                    alpha = resize_long_edge_array(alpha, target)

            target_dtype = eff_profile.target_dtype(dtypes_in[slot])
            arr_int = float_to_dtype_array(
                out,
                target_dtype,
                alpha,
                base_channels_list[slot],
                float_normalisation=norms[slot],
            )

            if dry_run:
                _tick()
                continue

            dst = destinations[idx]
            with ProcessingContext(dst) as staged:
                # No per-file metadata dict is passed here; tags are reconstructed by save path/ICC.
                save_image(
                    staged,
                    arr_int,
                    target_dtype,
                    metadata=None,
                    icc_profile=iccs[slot],
                    compression=compression,
                )
            processed += 1
            _tick()

        # free bucket arrays
        del arrays, batch, out_batch

    return processed


__all__ = [
    "_PROGRESS_WRAPPER",
    "_coerce_resize_target",
    "_wrap_with_progress",
    "collect_images",
    "ensure_output_path",
    "process_image",
    "process_single_image",
    "resize_array",
    "resize_bilinear",
    "resize_long_edge_array",
    "process_images_batch",
]