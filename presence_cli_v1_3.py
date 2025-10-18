# path: presence_cli_v1_3.py
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import typer
from PIL import Image, ImageOps

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Presence v1.3 CLI utilities.")

# Optional OpenCV; degrade gracefully when not present.
try:
    import cv2  # type: ignore

    _CV_OK = True
except Exception:
    cv2 = None  # type: ignore
    _CV_OK = False


@dataclass
class MeasureResult:
    image: str
    width: int
    height: int
    aspect_input: str
    eye_line_pct: float
    gutters: Dict[str, int]
    confidence: float
    method: str


def _parse_aspect(aspect: str) -> Tuple[float, float]:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$", aspect)
    if not m:
        raise typer.BadParameter("Aspect must be like '4:5' or '1.91:1'")
    a_w = float(m.group(1))
    a_h = float(m.group(2))
    if a_w <= 0 or a_h <= 0:
        raise typer.BadParameter("Aspect components must be positive.")
    return a_w, a_h


def _pil_to_gray_np(im: Image.Image) -> np.ndarray:
    # why: EXIF orientation matters for composition/eyeline
    im = ImageOps.exif_transpose(im)
    return np.array(im.convert("L"), dtype=np.uint8)


def _detect_face_eyeline(gray: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (eye_y_px, confidence) using OpenCV Haar cascade; None on failure.

    why: Human eyeline ≈ 40–45% down from face bbox top; we use 42%.
    """
    if not _CV_OK:
        return None
    try:
        cascade_root = getattr(cv2.data, "haarcascades", "")
        cascade_path = cascade_root + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return None
        eq = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            eq, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        eye_y = float(y) + 0.42 * float(h)

        # Confidence: upweight larger faces (relative to frame height), clamp 0..1
        rel = min(1.0, float(h) / float(gray.shape[0] + 1e-6))
        conf = min(1.0, 0.7 + 0.3 * rel)
        return eye_y, conf
    except Exception:
        return None


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Vectorized 3×3 Sobel magnitude on reflect-padded image."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    p = np.pad(gray.astype(np.float32), 1, mode="reflect")

    # Slices for each kernel tap (avoid Python loops for speed/correctness)
    a = p[:-2, :-2]
    b = p[:-2, 1:-1]
    c = p[:-2, 2:]
    d = p[1:-1, :-2]
    e = p[1:-1, 1:-1]
    f = p[1:-1, 2:]
    g = p[2:, :-2]
    h = p[2:, 1:-1]
    i = p[2:, 2:]

    gx = (
        a * kx[0, 0]
        + b * kx[0, 1]
        + c * kx[0, 2]
        + d * kx[1, 0]
        + e * kx[1, 1]
        + f * kx[1, 2]
        + g * kx[2, 0]
        + h * kx[2, 1]
        + i * kx[2, 2]
    )
    gy = (
        a * ky[0, 0]
        + b * ky[0, 1]
        + c * ky[0, 2]
        + d * ky[1, 0]
        + e * ky[1, 1]
        + f * ky[1, 2]
        + g * ky[2, 0]
        + h * ky[2, 1]
        + i * ky[2, 2]
    )
    return np.hypot(gx, gy)


def _edge_fallback_eyeline(gray: np.ndarray) -> Tuple[float, float]:
    """Edge-energy fallback for eyeline without OpenCV.

    why: Portrait eyelines typically sit within top ~65% of frame.
    """
    mag = _sobel_magnitude(gray)
    upto = max(1, int(gray.shape[0] * 0.65))
    row_energy = mag[:upto, :].mean(axis=1)
    idx = int(row_energy.argmax())
    eye_y = float(idx)
    conf = 0.6
    return eye_y, conf


def _compute_gutters(w: int, h: int, a_w: float, a_h: float) -> Dict[str, int]:
    """Compute letter/pillarbox gutters to reach aspect a_w:a_h (ints, ≥0)."""
    target_w = h * (a_w / a_h)
    if target_w > w + 1e-6:
        pad = target_w - w
        half = int(round(pad / 2.0))
        return {"left_px": max(0, half), "right_px": max(0, half), "top_px": 0, "bottom_px": 0}

    target_h = w * (a_h / a_w)
    if target_h > h + 1e-6:
        pad = target_h - h
        half = int(round(pad / 2.0))
        return {"left_px": 0, "right_px": 0, "top_px": max(0, half), "bottom_px": max(0, half)}

    return {"left_px": 0, "right_px": 0, "top_px": 0, "bottom_px": 0}


def _confidence_adjust(conf: float, gray: np.ndarray) -> float:
    """Reduce confidence for tiny or low-contrast images."""
    H, W = gray.shape
    if min(H, W) < 512:
        conf -= 0.1
    if float(gray.std()) < 8.0:
        conf -= 0.1
    return max(0.0, min(1.0, conf))


@app.command("measure")
def measure(
    image: Path = typer.Option(
        ..., "--image", "-i", exists=True, file_okay=True, dir_okay=False, readable=True
    ),
    aspect: str = typer.Option("4:5", "--aspect", "-a", help="Target aspect ratio (e.g., 4:5, 1.91:1)"),
) -> None:
    """
    Auto-measure eyeline (% of image height) and gutters to letter/pillarbox to the given aspect.

    Prints JSON: {"eye_line_pct","gutters","confidence","width","height","method",...}
    """
    a_w, a_h = _parse_aspect(aspect)

    try:
        with Image.open(image) as im:
            gray = _pil_to_gray_np(im)
            w, h = im.size  # after EXIF transpose, size matches pixel array orientation
    except Exception as e:
        typer.secho(f"Failed to open image: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    method = "edge"
    res = _detect_face_eyeline(gray)
    if res is not None:
        eye_y, conf = res
        method = "face"
    else:
        eye_y, conf = _edge_fallback_eyeline(gray)

    conf = _confidence_adjust(conf, gray)
    eye_line_pct = float(np.clip((eye_y / float(h)) * 100.0, 0.0, 100.0))
    gutters = _compute_gutters(w, h, a_w, a_h)

    out = MeasureResult(
        image=str(image),
        width=w,
        height=h,
        aspect_input=f"{a_w:g}:{a_h:g}",
        eye_line_pct=round(eye_line_pct, 2),
        gutters=gutters,
        confidence=round(conf, 2),
        method=method,
    )
    print(json.dumps(asdict(out), ensure_ascii=False, separators=(",", ":")))


def main() -> None:
    app()


if __name__ == "__main__":
    main()