# path: luxury_tiff_batch_processor/cli.py
"""Typer-based batch CLI for luxury TIFF processing.

Install an entrypoint like:
    lux-batch = luxury_tiff_batch_processor.cli:main
"""
from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, TYPE_CHECKING, cast

import typer

from .adjustments import (
    AdjustmentSettings,
    LUXURY_PRESETS,
    apply_adjustments,
    batch_apply_adjustments,
)
from .io_utils import image_to_float, save_image

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # noqa: F401
    from numpy.typing import NDArray  # noqa: F401

try:  # optional YAML support
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

AppMethod = Literal["auto", "vectorized", "multiprocessing"]

app = typer.Typer(name="lux-batch", no_args_is_help=True, add_completion=False)

# --------------------------- internal helpers (why: robustness) ---------------

def _ensure_mutually_exclusive(preset: Optional[str], preset_file: Optional[Path]) -> None:
    if preset and preset_file:
        raise typer.BadParameter("Use either --preset or --preset-file, not both.")
    if not preset and not preset_file:
        raise typer.BadParameter("One of --preset or --preset-file is required.")

def _is_tiff(p: Path) -> bool:
    return p.suffix.lower() in {".tif", ".tiff"}

def _collect_inputs(sources: Sequence[Path]) -> List[Tuple[Path, Path]]:
    """Return list of (input_path, anchor_root). Anchor is used for mirroring."""
    pairs: List[Tuple[Path, Path]] = []
    for src in sources:
        if src.is_file() and _is_tiff(src):
            pairs.append((src.resolve(), src.parent.resolve()))
        elif src.is_dir():
            for p in src.rglob("*"):
                if p.is_file() and _is_tiff(p):
                    pairs.append((p.resolve(), src.resolve()))
    return pairs

def _suffix_for_outputs(uniform_name: Optional[str]) -> str:
    return (uniform_name or "lux").strip().replace(" ", "_")

def _build_output_path(out_root: Path, inp: Path, anchor: Path, suffix: str) -> Path:
    rel = inp.relative_to(anchor)
    new_name = f"{rel.stem}_{suffix}{rel.suffix}"
    return (out_root / rel.parent / new_name).resolve()

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _extract_array(image_result) -> "NDArray[np.float32]":
    # Why: tolerate minor IO result-shape variations without breaking.
    import numpy as np  # lazy import
    for attr in ("array", "arr", "data"):
        if hasattr(image_result, attr):
            return getattr(image_result, attr).astype("float32", copy=False)
    if isinstance(image_result, tuple) and image_result:
        return image_result[0].astype("float32", copy=False)
    if isinstance(image_result, dict) and "array" in image_result:
        return image_result["array"].astype("float32", copy=False)
    raise RuntimeError("Unsupported ImageToFloatResult; expected an '.array' field.")

def _safe_save(path: Path, array, reference) -> None:
    # Why: preserve metadata when available; degrade gracefully if signature changes.
    try:
        save_image(path, array, reference=reference)  # type: ignore[call-arg]
    except TypeError:
        try:
            save_image(path, array, reference)  # type: ignore[misc]
        except TypeError:
            save_image(path, array)  # type: ignore[misc]

def _settings_from_dict(d: Dict[str, object]) -> AdjustmentSettings:
    valid = {f.name for f in fields(AdjustmentSettings)}
    filtered = {k: v for k, v in d.items() if k in valid}
    # Cast values to appropriate types - mypy needs this for **kwargs unpacking
    return AdjustmentSettings(**filtered)  # type: ignore[arg-type]

def _load_preset_file(
    preset_file: Path,
    inputs: Sequence[Tuple[Path, Path]],
) -> Union[AdjustmentSettings, List[AdjustmentSettings]]:
    """Supports:
    - Single dict of settings (optionally {'preset': 'name', ...overrides})
    - List[settings] with len==N
    - {'by_name': {'file.tif': {...}}, 'default': {...}} mapping by basename
    YAML requires PyYAML if .yml/.yaml is used.
    """
    text = preset_file.read_text(encoding="utf-8")
    if preset_file.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise typer.BadParameter("PyYAML required for YAML preset files.")
        data = yaml.safe_load(text)  # type: ignore
    else:
        data = json.loads(text)

    if isinstance(data, dict):
        if "preset" in data:
            base_name = str(data["preset"])
            if base_name not in LUXURY_PRESETS:
                raise typer.BadParameter(f"Unknown preset in file: {base_name!r}")
            base = LUXURY_PRESETS[base_name]
            overrides = {k: v for k, v in data.items() if k != "preset"}
            return _settings_from_dict({**base.__dict__, **overrides})

        if "by_name" in data:
            by_name = data.get("by_name", {}) or {}
            default = data.get("default", None)
            default_settings = _settings_from_dict(default) if isinstance(default, dict) else None
            out: List[AdjustmentSettings] = []
            for inp, _anchor in inputs:
                spec = by_name.get(inp.name)
                if spec is None:
                    if default_settings is None:
                        raise typer.BadParameter(f"Missing settings for {inp.name!r} and no default provided.")
                    out.append(default_settings)
                else:
                    out.append(_settings_from_dict(spec))
            return out

        return _settings_from_dict(data)

    if isinstance(data, list):
        out = [_settings_from_dict(x) for x in data if isinstance(x, dict)]
        if len(out) != len(inputs):
            raise typer.BadParameter(f"Preset list length {len(out)} != number of inputs {len(inputs)}.")
        return out

    raise typer.BadParameter("Unrecognized preset-file schema.")

# --------------------------- top-level worker (picklable) ----------------------

def _worker_task(
    idx: int,
    inputs: Sequence[Tuple[Path, Path]],
    outputs: Sequence[Path],
    overwrite: bool,
    single_settings: Sequence[AdjustmentSettings],
) -> Tuple[int, Optional[str]]:
    """Process one image entry; returns (index, error-message-or-None)."""
    p_in, _anchor = inputs[idx]
    p_out = outputs[idx]
    if p_out.exists() and not overwrite:
        return idx, None
    try:
        res = image_to_float(str(p_in), return_format="object")
        arr = _extract_array(res)
        out = apply_adjustments(arr, single_settings[idx])
        _safe_save(p_out, out, res)
        return idx, None
    except Exception as exc:  # keep other items running
        return idx, f"{p_in} → {exc}"

# --------------------------------- CLI ---------------------------------------

@app.command("run")
def lux_batch(
    sources: List[Path] = typer.Argument(..., exists=True, readable=True, help="Files and/or directories to process."),
    out_dir: Path = typer.Option(..., "--out-dir", "-o", help="Output root; input tree is mirrored underneath."),
    preset: Optional[str] = typer.Option(None, "--preset", help=f"Name in presets: {', '.join(sorted(LUXURY_PRESETS))}"),
    preset_file: Optional[Path] = typer.Option(None, "--preset-file", exists=True, readable=True,
                                               help="JSON/YAML with a single settings dict, list, or by_name map."),
    method: AppMethod = typer.Option("auto", "--method", case_sensitive=False,
                                     help="auto | vectorized | multiprocessing"),
    workers: Optional[int] = typer.Option(None, "--workers", min=1, help="Process count for multiprocessing."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs."),
) -> None:
    """Process TIFFs, mirroring the input directory structure."""
    _ensure_mutually_exclusive(preset, preset_file)

    inputs = _collect_inputs(sources)
    if not inputs:
        typer.secho("No input .tif/.tiff files found.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    out_dir.mkdir(parents=True, exist_ok=True)

    if preset:
        if preset not in LUXURY_PRESETS:
            raise typer.BadParameter(f"Unknown preset: {preset!r}")
        settings: Union[AdjustmentSettings, List[AdjustmentSettings]] = LUXURY_PRESETS[preset]
        suffix = _suffix_for_outputs(preset)
    else:
        settings = _load_preset_file(preset_file, inputs)  # type: ignore[arg-type]
        suffix = _suffix_for_outputs(None if isinstance(settings, list) else "lux")

    outputs: List[Path] = []
    for inp, anchor in inputs:
        out_path = _build_output_path(out_dir, inp, anchor, suffix)
        _ensure_parent(out_path)
        if out_path.exists() and not overwrite:
            typer.secho(f"Skip existing: {out_path}", fg=typer.colors.YELLOW)
        outputs.append(out_path)

    failures = 0

    if isinstance(settings, AdjustmentSettings):
        # Single settings: vectorize per image size; single read pass.
        from collections import defaultdict
        import numpy as np  # lazy import

        buckets: Dict[Tuple[int, int], List[Tuple[int, "NDArray[np.float32]", object]]] = defaultdict(list)
        for idx, (inp, _anchor) in enumerate(inputs):
            res = image_to_float(str(inp), return_format="object")
            arr = _extract_array(res)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                failures += 1
                typer.secho(f"Skip non-RGB 3-channel image: {inp}", fg=typer.colors.RED)
                continue
            h, w = int(arr.shape[0]), int(arr.shape[1])
            buckets[(h, w)].append((idx, arr, res))

        for shape, items in buckets.items():
            inds, arrays, refs = zip(*items)
            batch = np.stack(arrays, axis=0).astype("float32", copy=False)
            try:
                out_batch = batch_apply_adjustments(batch, settings, method=method, workers=workers)
            except Exception as e:
                typer.secho(f"[vectorized->serial fallback] {shape}: {e}", fg=typer.colors.YELLOW)
                out_batch = None

            if out_batch is None:
                processed = [apply_adjustments(a, settings) for a in arrays]
            else:
                processed = [out_batch[i] for i in range(out_batch.shape[0])]

            for slot, idx in enumerate(inds):
                try:
                    if outputs[idx].exists() and not overwrite:
                        continue
                    _safe_save(outputs[idx], processed[slot], refs[slot])
                    typer.echo(str(outputs[idx]))
                except Exception as e:
                    failures += 1
                    typer.secho(f"Failed: {inputs[idx][0]} → {e}", fg=typer.colors.RED)

            # free bucket memory
            del arrays, refs, processed, out_batch

    else:
        # Per-file settings: allow multiprocessing; each worker loads/saves its own file.
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n = len(inputs)
        use_mp = (method == "multiprocessing") or (method == "auto" and n >= 8)
        if use_mp:
            max_workers = workers or max(1, (os.cpu_count() or 1))
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_worker_task, i, inputs, outputs, overwrite, settings) for i in range(n)]
                for fut in as_completed(futs):
                    idx, err = fut.result()
                    if err:
                        failures += 1
                        typer.secho(f"Failed: {err}", fg=typer.colors.RED)
                    else:
                        typer.echo(str(outputs[idx]))
        else:
            for i in range(n):
                idx, err = _worker_task(i, inputs, outputs, overwrite, settings)
                if err:
                    failures += 1
                    typer.secho(f"Failed: {err}", fg=typer.colors.RED)
                else:
                    typer.echo(str(outputs[idx]))

    if failures:
        raise typer.Exit(code=1)

def main() -> None:
    app()

if __name__ == "__main__":
    main()
