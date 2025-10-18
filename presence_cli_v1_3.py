# path: presence_cli_v1_3.py
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import typer
from PIL import Image

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Presence v1.3 CLI utilities.")

# Optional OpenCV; we degrade gracefully when not present.
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
    gutters: dict
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
    arr = np.array(im.convert("L"), dtype=np.uint8)
    return arr


def _detect_face_eyeline(gray: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (eye_y_px, confidence) if OpenCV Haar cascade can find a face.

    Eye line approximated ~42% down from face top.
    """
    if not _CV_OK:
        return None
    try:
        cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return None
        eq = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(eq, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) == 0:
            return None
        # Largest face by area
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        eye_y = float(y) + 0.42 * float(h)  # typical eyes ~0.4–0.45 of face height
        conf = 0.9
        return eye_y, conf
    except Exception:
        return None


def _edge_fallback_eyeline(gray: np.ndarray) -> Tuple[float, float]:
    """Edge-energy fallback for eye-line without OpenCV.

    Focus on upper 65%; pick row with max edge magnitude.
    """
    # Sobel gradients
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    # Minimal Sobel kernels
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    # Convolve (naive valid conv via slicing for speed and zero-copy)
    pad = 1
    padded = np.pad(gray.astype(np.float32), pad, mode="reflect")
    H, W = gray.shape
    out_x = np.empty_like(gray, dtype=np.float32)
    out_y = np.empty_like(gray, dtype=np.float32)
    for i in range(H):
        r = padded[i:i+3, :]
        out_x[i, :] = (r[:, 0:W] * kx[0, 0] + r[:, 1:W+1] * kx[0, 1] + r[:, 2:W+2] * kx[0, 2] +
                       padded[i+1, 0:W] * kx[1, 0] + padded[i+1, 1:W+1] * kx[1, 1] + padded[i+1, 2:W+2] * kx[1, 2] +
                       padded[i+2, 0:W] * kx[2, 0] + padded[i+2, 1:W+1] * kx[2, 1] + padded[i+2, 2:W+2] * kx[2, 2])
        out_y[i, :] = (r[:, 0:W] * ky[0, 0] + r[:, 1:W+1] * ky[0, 1] + r[:, 2:W+2] * ky[0, 2] +
                       padded[i+1, 0:W] * ky[1, 0] + padded[i+1, 1:W+1] * ky[1, 1] + padded[i+1, 2:W+2] * ky[1, 2] +
                       padded[i+2, 0:W] * ky[2, 0] + padded[i+2, 1:W+1] * ky[2, 1] + padded[i+2, 2:W+2] * ky[2, 2])
    mag = np.hypot(out_x, out_y)
    # Limit to upper 65% (where eyes typically sit in portraits)
    upto = max(1, int(gray.shape[0] * 0.65))
    row_energy = mag[:upto, :].mean(axis=1)
    idx = int(row_energy.argmax())
    eye_y = float(idx)
    # Confidence heuristic: sharper edges + higher variance -> better
    conf = 0.6
    return eye_y, conf


def _compute_gutters(w: int, h: int, a_w: float, a_h: float) -> dict:
    """Compute letterbox gutters to reach aspect a_w:a_h (integers, >=0)."""
    target_w = h * (a_w / a_h)
    if target_w > w + 1e-6:
        pad = target_w - w
        left = right = max(0, int(round(pad / 2.0)))
        return {"left_px": left, "right_px": right, "top_px": 0, "bottom_px": 0}
    # vertical bars
    target_h = w * (a_h / a_w)
    if target_h > h + 1e-6:
        pad = target_h - h
        top = bottom = max(0, int(round(pad / 2.0)))
        return {"left_px": 0, "right_px": 0, "top_px": top, "bottom_px": bottom}
    return {"left_px": 0, "right_px": 0, "top_px": 0, "bottom_px": 0}


def _confidence_adjust(conf: float, gray: np.ndarray) -> float:
    """Reduce confidence for tiny or flat images."""
    H, W = gray.shape
    if min(H, W) < 512:
        conf -= 0.1
    std = float(gray.std())
    if std < 8.0:
        conf -= 0.1
    return max(0.0, min(1.0, conf))


@app.command("measure")
def measure(
    image: Path = typer.Option(..., "--image", "-i", exists=True, file_okay=True, dir_okay=False, readable=True),
    aspect: str = typer.Option("4:5", "--aspect", "-a", help="Target aspect ratio (e.g., 4:5, 1.91:1)"),
) -> None:
    """
    Auto-measure eye-line (as a percent of image height) and gutters to letterbox to the given aspect.

    Prints a JSON report: {"eye_line_pct", "gutters", "confidence", "width", "height", "method"}.
    """
    a_w, a_h = _parse_aspect(aspect)

    with Image.open(image) as im:
        w, h = im.size
        gray = _pil_to_gray_np(im)

    # Try face-based method first
    method = "fallback"
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