# file: picacho_lane_luts/process_renderings_750.py
from __future__ import annotations

import argparse
import concurrent.futures as futures
import fnmatch
import importlib
import importlib.util
import json
import math
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif"}
CONVERTIBLE_IMAGE_SUFFIXES = {".tif", ".tiff", ".webp", ".bmp", ".tga", ".psd", ".exr"}

@dataclass(frozen=True)
class RenderRecipe:
    highlight_compression: float = 0.0
    shadow_lift: float = 0.0
    temperature_shift: float = 0.0
    highlight_cool: float = 0.0
    shadow_warm: float = 0.0
    saturation: float = 1.0
    contrast: float = 1.0
    clarity: float = 0.0
    sharpness: float = 1.0
    contact_shadow: float = 0.0
    contact_radius: int = 5
    haze: float = 0.0

BASE_RECIPES: Dict[str, RenderRecipe] = {
    "aerial": RenderRecipe(
        highlight_compression=0.35, temperature_shift=0.06, saturation=1.02, clarity=0.1, contact_shadow=0.06, haze=0.05
    ),
    "greatroom": RenderRecipe(
        highlight_compression=0.15, shadow_lift=0.05, temperature_shift=-0.03, highlight_cool=0.08, shadow_warm=0.04,
        saturation=1.1, contrast=1.05, clarity=0.18, sharpness=1.05, contact_shadow=0.08
    ),
    "kitchen": RenderRecipe(
        highlight_compression=0.12, shadow_lift=0.04, temperature_shift=-0.05, highlight_cool=0.07, shadow_warm=0.05,
        saturation=1.08, contrast=1.07, clarity=0.22, sharpness=1.18, contact_shadow=0.12
    ),
    "pool": RenderRecipe(
        highlight_compression=0.08, temperature_shift=0.1, saturation=1.05, contrast=1.03, clarity=0.12,
        sharpness=1.04, contact_shadow=0.1, contact_radius=6
    ),
    "primarybedroom": RenderRecipe(
        highlight_compression=0.1, shadow_lift=0.2, temperature_shift=-0.04, highlight_cool=0.1, shadow_warm=0.03,
        saturation=1.06, contrast=1.04, clarity=0.16, sharpness=1.06, contact_shadow=0.09, contact_radius=7
    ),
}

def _iter_render_candidates(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file():
            yield path

def _require_module(module_name: str):
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(f"The optional dependency '{module_name}' is required to convert that asset.")
    return importlib.import_module(module_name)

def _convert_exr_to_jpg(exr_path: Path, jpg_path: Path, *, exposure: float = 1.0, gamma: float = 2.2) -> None:
    OpenEXR = _require_module("OpenEXR")
    Imath = _require_module("Imath")
    exr_file = OpenEXR.InputFile(str(exr_path))
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32) for c in ("R", "G", "B")]
    exr_file.close()
    rgb_channels = [ch.reshape(height, width) for ch in channels]
    rgb = np.stack(rgb_channels, axis=-1)
    x = np.clip(rgb * float(exposure), 0.0, None)
    tonemapped = x / (1.0 + x)
    tonemapped = np.power(np.clip(tonemapped, 0.0, 1.0), 1.0 / float(gamma))
    out8 = (tonemapped * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out8, mode="RGB").save(jpg_path, format="JPEG", quality=95)

def _convert_with_pillow(source: Path, destination: Path) -> None:
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in {"RGBA", "LA", "P"}:
            base = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            base.paste(img, mask=img.split()[-1])
            img = base
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(destination, format="JPEG", quality=95)

def convert_renderings_to_jpeg(input_dir: Path, output_dir: Path | None = None) -> Path:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "converted_for_api"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in _iter_render_candidates(input_dir):
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            copied = output_dir / path.name
            if copied.exists() and copied.stat().st_mtime >= path.stat().st_mtime:
                continue
            shutil.copy2(path, copied)
            continue
        if suffix not in CONVERTIBLE_IMAGE_SUFFIXES:
            continue
        destination = output_dir / (path.stem + ".jpg")
        if destination.exists() and path.stat().st_mtime <= destination.stat().st_mtime:
            continue
        if suffix == ".exr":
            _convert_exr_to_jpg(path, destination)
        else:
            _convert_with_pillow(path, destination)
    return output_dir

def ensure_supported_renderings(input_dir: Path) -> Path:
    paths = list(_iter_render_candidates(input_dir))
    if any(p.suffix.lower() in CONVERTIBLE_IMAGE_SUFFIXES for p in paths):
        return convert_renderings_to_jpeg(input_dir)
    return input_dir

def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def _save_rgb(arr: np.ndarray, path: Path, *, quality: int = 95) -> None:
    clipped = np.clip(arr, 0.0, 1.0)
    out_im = Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        out_im.save(path, format="JPEG", quality=quality, subsampling=1, optimize=True)
    else:
        out_im.save(path)

def _luminance(arr: np.ndarray) -> np.ndarray:
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

def _apply_highlight_compression(arr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return arr
    return np.power(np.clip(arr, 0.0, 1.0), 1.0 + float(strength))

def _apply_shadow_lift(arr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return arr
    return np.power(np.clip(arr, 0.0, 1.0), 1.0 / (1.0 + float(strength)))

def _apply_split_tone(arr: np.ndarray, *, highlight_cool: float, shadow_warm: float) -> None:
    if highlight_cool <= 0 and shadow_warm <= 0:
        return
    lum = _luminance(arr)
    if highlight_cool > 0:
        hi = np.clip((lum - 0.55) / 0.4, 0.0, 1.0)[..., None]
        arr[..., 0] *= 1.0 - highlight_cool * hi
        arr[..., 2] *= 1.0 + highlight_cool * hi
    if shadow_warm > 0:
        lo = np.clip((0.4 - lum) / 0.4, 0.0, 1.0)[..., None]
        arr[..., 0] *= 1.0 + shadow_warm * lo
        arr[..., 2] *= 1.0 - shadow_warm * lo

def _apply_temperature(arr: np.ndarray, shift: float) -> np.ndarray:
    if shift == 0.0:
        return arr
    arr[..., 0] *= 1.0 + shift
    arr[..., 2] *= 1.0 - shift
    return arr

def _apply_contact_shadows(arr: np.ndarray, strength: float, radius: int) -> np.ndarray:
    if strength <= 0.0:
        return arr
    lum = _luminance(arr)
    pil_lum = Image.fromarray((np.clip(lum, 0.0, 1.0) * 255.0).astype(np.uint8))
    blurred = np.asarray(pil_lum.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0
    ao = np.clip(blurred - lum, 0.0, 1.0)[..., None]
    arr *= 1.0 - strength * ao
    return arr

def _apply_haze(arr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return arr
    lum = _luminance(arr)[..., None]
    haze = np.clip(1.0 - lum, 0.0, 1.0)
    return np.clip(arr * (1.0 - amount) + haze * amount, 0.0, 1.0)

def _apply_saturation(pil_img: Image.Image, factor: float) -> Image.Image:
    if abs(factor - 1.0) < 1e-3:
        return pil_img
    return ImageEnhance.Color(pil_img).enhance(factor)

def _apply_contrast(pil_img: Image.Image, factor: float) -> Image.Image:
    if abs(factor - 1.0) < 1e-3:
        return pil_img
    return ImageEnhance.Contrast(pil_img).enhance(factor)

def _apply_clarity(pil_img: Image.Image, amount: float) -> Image.Image:
    if amount <= 0.0:
        return pil_img
    percent = int(150 * amount)
    return pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))

def _apply_sharpness(pil_img: Image.Image, factor: float) -> Image.Image:
    if abs(factor - 1.0) < 1e-3:
        return pil_img
    return ImageEnhance.Sharpness(pil_img).enhance(factor)

def _match_recipe(path: Path) -> RenderRecipe:
    stem = path.stem.lower()
    for key, recipe in BASE_RECIPES.items():
        if key in stem:
            return recipe
    raise KeyError(f"Unable to match render '{path.name}' to a configured recipe")

@dataclass
class Job:
    src: Path
    dst: Path
    recipe: RenderRecipe

@dataclass
class JobResult:
    src: str
    dst: str
    status: Literal["ok", "skipped", "error"]
    detail: str = ""

def process_render(path: Path, output_path: Path, recipe: RenderRecipe, *, quality: int) -> None:
    arr = _load_rgb(path)
    arr = _apply_highlight_compression(arr, recipe.highlight_compression)
    arr = _apply_shadow_lift(arr, recipe.shadow_lift)
    _apply_temperature(arr, recipe.temperature_shift)
    _apply_split_tone(arr, highlight_cool=recipe.highlight_cool, shadow_warm=recipe.shadow_warm)
    _apply_contact_shadows(arr, recipe.contact_shadow, recipe.contact_radius)
    arr = _apply_haze(arr, recipe.haze)
    pil_img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    pil_img = _apply_saturation(pil_img, recipe.saturation)
    pil_img = _apply_contrast(pil_img, recipe.contrast)
    pil_img = _apply_clarity(pil_img, recipe.clarity)
    pil_img = _apply_sharpness(pil_img, recipe.sharpness)
    final = np.asarray(pil_img, dtype=np.float32) / 255.0
    _save_rgb(final, output_path, quality=quality)

def _plan_jobs(input_dir: Path, output_dir: Path, patterns: Sequence[str]) -> List[Job]:
    normalized_input = ensure_supported_renderings(input_dir)
    jobs: List[Job] = []
    for path in sorted(Path(normalized_input).glob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (SUPPORTED_IMAGE_SUFFIXES | CONVERTIBLE_IMAGE_SUFFIXES):
            continue
        if patterns and not any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
            continue
        try:
            recipe = _match_recipe(path)
        except KeyError:
            continue
        out_name = f"{path.stem}_graded{path.suffix.lower()}"
        jobs.append(Job(src=path, dst=(output_dir / out_name), recipe=recipe))
    return jobs

def _should_skip(dst: Path, src: Path, overwrite: bool) -> bool:
    if overwrite:
        return False
    return dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime

def _run_job(job: Job, *, quality: int, overwrite: bool, on_error: Literal["skip", "raise"]) -> JobResult:
    try:
        job.dst.parent.mkdir(parents=True, exist_ok=True)
        if _should_skip(job.dst, job.src, overwrite):
            return JobResult(src=str(job.src), dst=str(job.dst), status="skipped", detail="up-to-date")
        process_render(job.src, job.dst, job.recipe, quality=quality)
        return JobResult(src=str(job.src), dst=str(job.dst), status="ok")
    except Exception as e:
        if on_error == "raise":
            raise
        return JobResult(src=str(job.src), dst=str(job.dst), status="error", detail=str(e))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process the 750px rendering review set.")
    p.add_argument("--input", type=Path, default=Path("/Users/rc/input_renderings_750"), help="Input directory.")
    p.add_argument("--output", type=Path, default=Path("/Users/rc/output_renderings_750"), help="Output directory.")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Parallel jobs. Use 1 for serial.")
    p.add_argument("--pattern", action="append", default=[], help="Glob filter; repeatable.")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if output is newer.")
    p.add_argument("--dry-run", action="store_true", help="Plan only; print summary and exit.")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality when saving .jpg.")
    p.add_argument("--on-error", choices=["skip", "raise"], default="skip", help="Behavior on processing error.")
    p.add_argument("--report", type=Path, default=None, help="Write a JSON report to this path.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = _plan_jobs(input_dir, output_dir, args.pattern)
    summary = {"input": str(input_dir), "output": str(output_dir), "planned": len(jobs), "jobs": [asdict(j) for j in jobs]}

    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, **summary}, indent=2))
        return

    results: List[JobResult] = []
    if max(1, int(args.jobs)) == 1:
        for j in jobs:
            r = _run_job(j, quality=args.quality, overwrite=args.overwrite, on_error=args.on_error)
            results.append(r)
            print(f"[{r.status}] {j.src.name} -> {j.dst.name}" + (f" ({r.detail})" if r.detail else ""))
    else:
        with futures.ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
            futs = [ex.submit(_run_job, j, quality=args.quality, overwrite=args.overwrite, on_error=args.on_error) for j in jobs]
            for fut, j in zip(futs, jobs):
                r = fut.result()
                results.append(r)
                print(f"[{r.status}] {j.src.name} -> {j.dst.name}" + (f" ({r.detail})" if r.detail else ""))

    ok = all(r.status in {"ok", "skipped"} for r in results)
    report = {"ok": ok, **summary, "results": [asdict(r) for r in results]}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()