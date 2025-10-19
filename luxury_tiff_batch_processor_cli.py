# file: luxury_tiff_batch_processor_cli.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preset-driven LUT grading pipeline orchestrated via FFmpeg.

Highlights
- Robust ffprobe probing + concise source summary.
- Smart tone-map planner (auto/off/operator) with HDR detection.
- Frame-rate assessment & conformance plan with tolerance.
- Filter-graph builder: denoise → tonemap → grade → LUT → effects → fps.
- Color metadata priority: explicit > from source > tone-map defaults > none.
- Fully runnable: includes minimal PRESETS registry.

Note: LUT files must exist at the specified paths or `--custom-lut` must be set.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Callable, NotRequired, TypedDict, Any, cast

# ---------------------------- Typed ffprobe data -------------------------------

class ProbeStream(TypedDict, total=False):
    codec_type: str
    codec_name: str
    width: int
    height: int
    avg_frame_rate: str
    r_frame_rate: str
    pix_fmt: str
    bits_per_raw_sample: str
    color_primaries: str
    color_trc: str
    color_space: str
    colorspace: str
    sample_rate: str
    channels: int
    tags: NotRequired[Dict[str, str]]

class ProbeFormat(TypedDict, total=False):
    format_name: str
    duration: str
    size: str
    bit_rate: str
    tags: NotRequired[Dict[str, str]]

class ProbeData(TypedDict, total=False):
    streams: List[ProbeStream]
    format: ProbeFormat

# ---------------------------- Data Models & Presets ----------------------------

@dataclass(frozen=True)
class Preset:
    """A grading preset with a human description and default config."""
    name: str
    description: str
    lut: Path
    defaults: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        out = dict(self.defaults)
        out["lut"] = self.lut
        return out


# Minimal, sensible presets; extend in real project.
_REPO_ROOT = Path(__file__).resolve().parent
_PRESET_LUTS = _REPO_ROOT / "luts"

PRESETS: Dict[str, Preset] = {
    "signature_estate": Preset(
        name="signature_estate",
        description="Warm, vibrant estate look with subtle clarity and fine grain.",
        lut=_PRESET_LUTS / "signature_estate.cube",
        defaults={
            "denoise": "soft",
            "contrast": 1.06,
            "saturation": 1.08,
            "gamma": 1.00,
            "brightness": 0.00,
            "warmth": 0.08,
            "cool": -0.02,
            "sharpen": "medium",
            "grain": 0.12,
            "deband": "off",
            "lut_strength": 1.0,
            "tone_map": "auto",
            "tone_map_peak": 1000.0,
            "tone_map_desat": 0.10,
        },
    ),
    "modern_minimal": Preset(
        name="modern_minimal",
        description="Neutral, clean, low-contrast commercial aesthetic.",
        lut=_PRESET_LUTS / "modern_minimal.cube",
        defaults={
            "denoise": "soft",
            "contrast": 0.98,
            "saturation": 0.96,
            "gamma": 1.02,
            "brightness": 0.02,
            "warmth": 0.00,
            "cool": 0.00,
            "sharpen": "soft",
            "grain": 0.06,
            "deband": "off",
            "lut_strength": 0.85,
            "tone_map": "auto",
            "tone_map_peak": 1000.0,
            "tone_map_desat": 0.08,
        },
    ),
    "heritage_cinematic": Preset(
        name="heritage_cinematic",
        description="Richer contrast, gentle warmth, light halation for cinematic feel.",
        lut=_PRESET_LUTS / "heritage_cinematic.cube",
        defaults={
            "denoise": "medium",
            "contrast": 1.12,
            "saturation": 1.05,
            "gamma": 0.98,
            "brightness": -0.01,
            "warmth": 0.10,
            "cool": -0.04,
            "sharpen": "soft",
            "grain": 0.18,
            "deband": "soft",
            "lut_strength": 0.95,
            "tone_map": "auto",
            "tone_map_peak": 1000.0,
            "tone_map_desat": 0.12,
            "halation_intensity": 0.15,
            "halation_radius": 20.0,
            "halation_threshold": 0.60,
        },
    ),
}

# --------------------------- Utility & Domain Types ----------------------------

@dataclass
class FrameRatePlan:
    """Summary of how the script will handle frame rate conformance."""
    target: Optional[str]
    note: str


HQDN3D_PRESETS = {
    "soft": "hqdn3d=luma_spatial=1.6:luma_tmp=3.2:chroma_spatial=1.2:chroma_tmp=2.8",
    "medium": "hqdn3d=luma_spatial=2.8:luma_tmp=4.5:chroma_spatial=2.0:chroma_tmp=4.0",
    "strong": "hqdn3d=luma_spatial=4.0:luma_tmp=6.5:chroma_spatial=3.0:chroma_tmp=5.0",
}

UNSHARP_PRESETS = {
    "soft": "unsharp=luma_msize_x=7:luma_msize_y=7:luma_amount=1.0:chroma_msize_x=5:chroma_msize_y=5:chroma_amount=0.4",
    "medium": "unsharp=luma_msize_x=7:luma_msize_y=7:luma_amount=1.35:chroma_msize_x=5:chroma_msize_y=5:chroma_amount=0.65",
    "strong": "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=1.6:chroma_msize_x=3:chroma_msize_y=3:chroma_amount=0.8",
}

DEBAND_PRESETS = {
    "soft": "gradfun=strength=0.45:radius=12",
    "medium": "gradfun=strength=0.70:radius=16",
    "strong": "gradfun=strength=0.90:radius=20",
}

@dataclass
class ToneMapPlan:
    """Description of the tone-mapping strategy for the current clip."""
    enabled: bool
    note: str
    config: Dict[str, object] = field(default_factory=dict)
    metadata: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None)


def list_presets() -> str:
    return "\n".join(f"- {key}: {preset.description}" for key, preset in PRESETS.items())


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ensure_tools_available() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil_which(tool):
            raise SystemExit(
                f"Required dependency '{tool}' was not found on PATH. Install FFmpeg to continue."
            )


def shutil_which(binary: str) -> Optional[str]:
    from shutil import which
    return which(binary)

# ------------------------------- Probe helpers --------------------------------

def probe_source(path: Path) -> ProbeData:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Unable to parse ffprobe output") from exc
    # Coerce to expected structure for typing
    streams = cast(List[ProbeStream], data.get("streams", []))
    fmt = cast(ProbeFormat, data.get("format", {}))
    return {"streams": streams, "format": fmt}

def _parse_probe_duration(raw: object) -> Optional[float]:
    """Return a finite float duration from ffprobe metadata when possible."""
    if raw in (None, ""):
        return None
    try:
        value = float(raw)  # accepts str/num
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value

# ------------------------------- Summarization --------------------------------

def summarize_probe(data: ProbeData) -> str:
    fmt = data.get("format", {})
    duration = fmt.get("duration")
    streams = data.get("streams", [])
    video: ProbeStream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio: ProbeStream = next((s for s in streams if s.get("codec_type") == "audio"), {})
    pieces: List[str] = []

    numeric_duration = _parse_probe_duration(duration)
    if numeric_duration is not None:
        pieces.append(f"duration {numeric_duration:.2f}s")

    if video:
        w = video.get("width")
        h = video.get("height")
        fps = describe_frame_rates(
            video.get("avg_frame_rate"), video.get("r_frame_rate")
        )
        codec = video.get("codec_name")
        pix_fmt = video.get("pix_fmt")

        video_info = f"video {codec} {w}x{h} @ {fps}"
        if pix_fmt:
            video_info += f" {pix_fmt}"

        bits_per_raw_sample = video.get("bits_per_raw_sample")
        if bits_per_raw_sample:
            video_info += f" {bits_per_raw_sample}bit"
        elif pix_fmt:
            if "10" in str(pix_fmt):
                video_info += " 10bit"
            elif "12" in str(pix_fmt):
                video_info += " 12bit"
            elif "16" in str(pix_fmt):
                video_info += " 16bit"

        color_parts: List[str] = []
        color_primaries = normalise_color_tag(video.get("color_primaries"))
        color_trc = normalise_color_tag(video.get("color_trc"))
        colorspace = normalise_color_tag(get_color_space_tag(video))

        if color_primaries:
            color_parts.append(f"primaries={color_primaries}")
        if color_trc:
            color_parts.append(f"trc={color_trc}")
        if colorspace:
            color_parts.append(f"space={colorspace}")

        if color_parts:
            video_info += f" ({', '.join(color_parts)})"

        pieces.append(video_info)

    if audio:
        codec = audio.get("codec_name")
        sr = audio.get("sample_rate")
        channels = audio.get("channels")
        pieces.append(f"audio {codec} {channels}ch {sr}Hz")

    return ", ".join(pieces)

def extract_video_stream(probe: ProbeData) -> ProbeStream:
    """Return the first video stream dictionary from an ffprobe result."""
    streams = probe.get("streams", [])
    return next((s for s in streams if s.get("codec_type") == "video"), {})

# --------------------------- HDR / Color Tag Utilities -------------------------

HDR_PRIMARIES = {"bt2020", "smpte432", "smpte431"}
HDR_TRANSFERS = {"smpte2084", "arib-std-b67", "hlg"}
HDR_MATRIX = {"bt2020nc", "bt2020ncl"}
INVALID_COLOR_TAGS = {"unknown", "unspecified", "undefined", "na"}

def normalise_color_tag(value: Optional[str]) -> Optional[str]:
    """Return a cleaned, lower-case color tag or ``None`` when not meaningful."""
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in INVALID_COLOR_TAGS:
        return None
    return lowered

def get_color_space_tag(stream: ProbeStream) -> Optional[str]:
    """Fetch the reported color space tag, handling legacy ffprobe key variants."""
    return stream.get("color_space") or stream.get("colorspace")

# --------------------------- Tone Mapping & Frame Rate -------------------------

def plan_tone_mapping(
    args: argparse.Namespace, probe: ProbeData
) -> ToneMapPlan:
    """Determine whether tone mapping should run for this clip."""
    method = (args.tone_map or "auto").lower()
    tone_map_peak = args.tone_map_peak
    tone_map_desat = args.tone_map_desat

    if method == "off":
        return ToneMapPlan(
            enabled=False, note="Tone mapping disabled by user preference."
        )

    video = extract_video_stream(probe)
    transfer = (video.get("color_trc") or "").lower()
    primaries = (video.get("color_primaries") or "").lower()
    matrix = (get_color_space_tag(video) or "").lower()

    hdr_indicators: List[str] = []
    if transfer in HDR_TRANSFERS:
        hdr_indicators.append(f"transfer={transfer}")
    if primaries in HDR_PRIMARIES:
        hdr_indicators.append(f"primaries={primaries}")
    if matrix in HDR_MATRIX:
        hdr_indicators.append(f"matrix={matrix}")

    detected_hdr = bool(hdr_indicators)
    if method == "auto" and not detected_hdr:
        return ToneMapPlan(
            enabled=False,
            note="Tone mapping not required; source tagged as SDR.",
        )

    chosen_method = method if method != "auto" else "hable"
    config: Dict[str, object] = {
        "tone_map": chosen_method,
        "tone_map_peak": tone_map_peak,
        "tone_map_desat": tone_map_desat,
    }
    note_parts = ["Applying HDR tone mapping using", chosen_method]
    if detected_hdr:
        note_parts.append(f"(detected {'/'.join(hdr_indicators)})")
    else:
        note_parts.append("(forced by user override)")

    metadata = ("bt709", "bt709", "bt709")
    return ToneMapPlan(
        enabled=True,
        note=" ".join(note_parts),
        config=config,
        metadata=metadata,
    )

def describe_frame_rates(avg: Optional[str], real: Optional[str]) -> str:
    """Return a friendly description of frame rate metadata."""
    avg_fraction = parse_ffprobe_fraction(avg)
    real_fraction = parse_ffprobe_fraction(real)
    if avg_fraction and real_fraction and avg_fraction != real_fraction:
        return (
            f"avg {format_fraction(avg_fraction)} ({float(avg_fraction):.3f}fps), "
            f"real {format_fraction(real_fraction)} ({float(real_fraction):.3f}fps)"
        )
    chosen = avg_fraction or real_fraction
    if chosen:
        return f"{format_fraction(chosen)} ({float(chosen):.3f}fps)"
    return "unknown"

STANDARD_FRAME_RATES: Tuple[Tuple[str, Fraction], ...] = (
    ("24000/1001", Fraction(24_000, 1_001)),
    ("24/1", Fraction(24, 1)),
    ("25/1", Fraction(25, 1)),
    ("30000/1001", Fraction(30_000, 1_001)),
    ("30/1", Fraction(30, 1)),
    ("50/1", Fraction(50, 1)),
    ("60000/1001", Fraction(60_000, 1_001)),
    ("60/1", Fraction(60, 1)),
)

def parse_ffprobe_fraction(value: Optional[str]) -> Optional[Fraction]:
    """Parse an FFprobe rational string into a Fraction."""
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        if "/" in value:
            return Fraction(value)
        return Fraction(float(value)).limit_denominator(100_000)
    except (ValueError, ZeroDivisionError):
        return None

def format_fraction(fraction: Fraction) -> str:
    return f"{fraction.numerator}/{fraction.denominator}"

def normalize_frame_rate(value: str) -> Tuple[str, Fraction]:
    """Normalise arbitrary frame-rate expressions to a rational string and Fraction."""
    parsed = parse_ffprobe_fraction(value)
    if not parsed:
        raise ValueError(f"Unable to parse frame rate value: {value}")
    reduced = parsed.limit_denominator(100_000)
    return format_fraction(reduced), reduced

def choose_standard_rate(fps: Fraction) -> Tuple[str, Fraction]:
    """Return the closest known delivery frame rate to the provided value."""
    best = min(
        STANDARD_FRAME_RATES,
        key=lambda item: abs(float(item[1]) - float(fps)),
    )
    return best[0], best[1]

def assess_frame_rate(
    probe: ProbeData,
    user_target: Optional[str],
    tolerance: float,
) -> FrameRatePlan:
    """Evaluate the source frame rate and decide whether to conform it."""
    tolerance = max(tolerance, 0.0001)
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        return FrameRatePlan(
            target=None,
            note="No video stream detected; skipping frame-rate adjustments.",
        )

    if user_target:
        normalized, fraction = normalize_frame_rate(user_target)
        return FrameRatePlan(
            target=normalized,
            note=(
                f"Frame-rate override requested; conforming output to {normalized}"
                f" ({float(fraction):.3f}fps)."
            ),
        )

    avg_fraction = parse_ffprobe_fraction(video.get("avg_frame_rate"))
    real_fraction = parse_ffprobe_fraction(video.get("r_frame_rate"))

    if not avg_fraction and not real_fraction:
        return FrameRatePlan(
            target=None,
            note="Source frame rate metadata unavailable; preserving stream timing.",
        )

    base = avg_fraction or real_fraction
    assert base is not None

    standard_label, standard_fraction = choose_standard_rate(base)
    std_diff = abs(float(standard_fraction) - float(base))

    vfr = False
    if avg_fraction and real_fraction:
        drift = abs(float(avg_fraction) - float(real_fraction))
        vfr = drift > tolerance
    else:
        drift = 0.0

    if vfr:
        return FrameRatePlan(
            target=standard_label,
            note=(
                "Detected variable frame-rate metadata ("
                f"avg {float(avg_fraction):.3f}fps vs real {float(real_fraction):.3f}fps); "
                f"conforming output to {standard_label} ({float(standard_fraction):.3f}fps)."
            ),
        )

    if std_diff > tolerance:
        return FrameRatePlan(
            target=standard_label,
            note=(
                f"Source frame rate {float(base):.3f}fps is off-standard by {std_diff:.3f}fps; "
                f"conforming to {standard_label} ({float(standard_fraction):.3f}fps) for smooth playback."
            ),
        )

    return FrameRatePlan(
        target=None,
        note=(
            f"Source frame rate {format_fraction(base)} ({float(base):.3f}fps) within tolerance; "
            "preserving timing."
        ),
    )

# -------------------------- Filter Graph Construction --------------------------

def build_filter_graph(config: Dict[str, object]) -> Tuple[str, str]:
    nodes: List[str] = []
    label_index = 0

    def next_label() -> str:
        nonlocal label_index
        label_index += 1
        return f"v{label_index}"

    current = "v0"
    nodes.append(f"[0:v]setsar=1[{current}]")

    # Promote to high bit-depth YUV before optional denoising.
    new_label = next_label()
    nodes.append(f"[{current}]format=yuv444p16le[{new_label}]")
    current = new_label

    tone_map = config.get("tone_map")
    if tone_map and str(tone_map).lower() != "off":
        tone_map_peak = config.get("tone_map_peak")
        pre_tonemap_args = ["transfer=linear"]
        if tone_map_peak is not None:
            pre_tonemap_args.append(f"npl={float(cast(Any, tone_map_peak)):.4f}")
        new_label = next_label()
        nodes.append(f"[{current}]zscale={':'.join(pre_tonemap_args)}[{new_label}]")
        current = new_label

        tonemap_args = [str(tone_map)]
        if tone_map_peak is not None:
            tonemap_args.append(f"peak={float(cast(Any, tone_map_peak)):.4f}")
        tone_map_desat = config.get("tone_map_desat")
        if tone_map_desat is not None:
            tonemap_args.append(f"desat={float(cast(Any, tone_map_desat)):.4f}")
        new_label = next_label()
        nodes.append(f"[{current}]tonemap={':'.join(tonemap_args)}[{new_label}]")
        current = new_label

        post_tonemap_args = [
            "primaries=bt709",
            "transfer=bt709",
            "matrix=bt709",
            "range=tv",
        ]
        new_label = next_label()
        nodes.append(f"[{current}]zscale={':'.join(post_tonemap_args)}[{new_label}]")
        current = new_label

    denoise = config.get("denoise")
    if denoise and str(denoise).lower() != "off":
        expr = HQDN3D_PRESETS.get(str(denoise).lower())
        if not expr:
            raise ValueError(f"Unsupported denoise preset: {denoise}")
        new_label = next_label()
        nodes.append(f"[{current}]{expr}[{new_label}]")
        current = new_label

    # Convert to planar RGB float for grading operations.
    new_label = next_label()
    nodes.append(f"[{current}]format=gbrpf32le[{new_label}]")
    current = new_label

    eq_parts: List[str] = []
    contrast = float(config.get("contrast", 1.0))
    saturation = float(config.get("saturation", 1.0))
    gamma = float(config.get("gamma", 1.0))
    brightness = float(config.get("brightness", 0.0))

    if not math.isclose(contrast, 1.0, abs_tol=1e-3):
        eq_parts.append(f"contrast={contrast:.4f}")
    if not math.isclose(saturation, 1.0, abs_tol=1e-3):
        eq_parts.append(f"saturation={saturation:.4f}")
    if not math.isclose(gamma, 1.0, abs_tol=1e-3):
        eq_parts.append(f"gamma={gamma:.4f}")
    if not math.isclose(brightness, 0.0, abs_tol=1e-4):
        eq_parts.append(f"brightness={brightness:.4f}")

    post_eq_label = current
    if eq_parts:
        new_label = next_label()
        nodes.append(f"[{current}]eq={':'.join(eq_parts)}[{new_label}]")
        current = new_label
    post_eq_label = current

    warmth = float(config.get("warmth", 0.0))
    cool = float(config.get("cool", 0.0))
    post_color_label = post_eq_label
    if not math.isclose(warmth, 0.0, abs_tol=1e-4) or not math.isclose(
        cool, 0.0, abs_tol=1e-4
    ):
        new_label = next_label()
        warmth_c = clamp(warmth, -0.5, 0.5)
        cool_c = clamp(cool, -0.5, 0.5)
        nodes.append(
            f"[{current}]colorbalance=rm={warmth_c:.4f}:gm=0.0000:bm={cool_c:.4f}[{new_label}]"
        )
        current = new_label
        post_color_label = current
    else:
        current = post_color_label

    pre_lut_label = post_color_label

    lut_path = Path(str(config.get("lut"))).resolve()
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    new_label = next_label()
    lut_expr = f"lut3d=file={shlex.quote(str(lut_path))}:interp=tetrahedral"
    nodes.append(f"[{current}]{lut_expr}[{new_label}]")
    current = new_label
    graded_label = current

    lut_strength = float(config.get("lut_strength", 1.0))
    if lut_strength < 0.999:
        blend_label = next_label()
        nodes.append(
            f"[{pre_lut_label}][{graded_label}]blend=all_expr='A*(1-{lut_strength:.4f})+B*{lut_strength:.4f}'[{blend_label}]"
        )
        current = blend_label

    sharpen = config.get("sharpen")
    if sharpen and str(sharpen).lower() != "off":
        expr = UNSHARP_PRESETS.get(str(sharpen).lower())
        if not expr:
            raise ValueError(f"Unsupported sharpen preset: {sharpen}")
        new_label = next_label()
        nodes.append(f"[{current}]{expr}[{new_label}]")
        current = new_label

    grain = float(config.get("grain", 0.0))
    if grain > 0.0:
        new_label = next_label()
        nodes.append(f"[{current}]noise=alls={grain:.2f}:allf=t+u[{new_label}]")
        current = new_label

    deband = config.get("deband")
    if deband and str(deband).lower() != "off":
        expr = DEBAND_PRESETS.get(str(deband).lower())
        if not expr:
            raise ValueError(f"Unsupported deband preset: {deband}")
        new_label = next_label()
        nodes.append(f"[{current}]{expr}[{new_label}]")
        current = new_label

    halation_intensity = float(config.get("halation_intensity", 0.0))
    if halation_intensity > 0.0:
        intensity = clamp(halation_intensity, 0.0, 1.0)
        radius = clamp(float(config.get("halation_radius", 18.0)), 0.0, 128.0)
        radius = max(radius, 1.0)
        threshold = clamp(float(config.get("halation_threshold", 0.6)), 0.0, 1.0)

        base_label = next_label()
        halo_label = next_label()
        nodes.append(f"[{current}]split=2[{base_label}][{halo_label}]")

        highlight_label = next_label()
        nodes.append(
            f"[{halo_label}]colorlevels=rimin={threshold:.3f}:gimin={threshold:.3f}:bimin={threshold:.3f}[{highlight_label}]"
        )

        blur_label = next_label()
        nodes.append(
            f"[{highlight_label}]gblur=sigma={radius:.2f}:steps=2[{blur_label}]"
        )

        tint_label = next_label()
        nodes.append(
            f"[{blur_label}]colorbalance=rm={intensity * 0.55:.4f}:gm={intensity * 0.25:.4f}:bm={-intensity * 0.15:.4f}[{tint_label}]"
        )

        blend_label = next_label()
        nodes.append(
            f"[{base_label}][{tint_label}]blend=all_expr='A+({intensity:.3f}*B)'[{blend_label}]"
        )
        current = blend_label

    target_fps = config.get("target_fps")
    if target_fps:
        new_label = next_label()
        nodes.append(f"[{current}]fps=fps={target_fps}[{new_label}]")
        current = new_label

    nodes.append(f"[{current}]format=yuv422p10le[vout]")
    graph = ";".join(nodes)
    return graph, "vout"

# ----------------------------- Color Tag Selection -----------------------------

def determine_color_metadata(
    args: argparse.Namespace, probe: ProbeData
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Determine color metadata based on priority: explicit > color-from-source > none."""
    if args.color_primaries or args.color_transfer or args.color_space:
        return args.color_primaries, args.color_transfer, args.color_space

    if args.color_from_source:
        video = extract_video_stream(probe)
        if video:
            primaries = normalise_color_tag(video.get("color_primaries"))
            transfer = normalise_color_tag(video.get("color_trc"))
            space = normalise_color_tag(get_color_space_tag(video))
            return primaries, transfer, space

    return None, None, None

# ------------------------------ Command Assembly -------------------------------

def build_command(
    input_path: Path,
    output_path: Path,
    filter_graph: str,
    filter_output: str,
    *,
    overwrite: bool,
    video_codec: str,
    prores_profile: int,
    bitrate: Optional[str],
    audio_codec: str,
    audio_bitrate: Optional[str],
    threads: Optional[int],
    log_level: str,
    preview_frames: Optional[int],
    vsync: str,
    color_primaries: Optional[str] = None,
    color_transfer: Optional[str] = None,
    color_space: Optional[str] = None,
) -> List[str]:
    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        log_level,
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-filter_complex",
        filter_graph,
        "-map",
        f"[{filter_output}]",
        "-map",
        "0:a?",
        "-c:v",
        video_codec,
    ]

    if video_codec == "prores_ks":
        cmd.extend(["-profile:v", str(prores_profile), "-pix_fmt", "yuv422p10le"])
    if bitrate:
        cmd.extend(["-b:v", bitrate])

    # Color metadata (explicit only; tone-map default handled earlier)
    if color_primaries:
        cmd.extend(["-color_primaries", color_primaries])
    if color_transfer:
        cmd.extend(["-color_trc", color_transfer])
    if color_space:
        cmd.extend(["-colorspace", color_space])

    cmd.extend(["-c:a", audio_codec])
    if audio_bitrate:
        cmd.extend(["-b:a", audio_bitrate])

    if preview_frames:
        cmd.extend(["-frames:v", str(preview_frames)])

    if threads:
        cmd.extend(["-threads", str(threads)])

    cmd.extend(["-vsync", vsync])
    cmd.append(str(output_path))
    return cmd

# ---------------------------------- CLI Layer ----------------------------------

class ListPresetsAction(argparse.Action):
    """Custom argparse action that prints presets and exits early."""

    def __init__(
        self,
        option_strings: List[str],
        dest: str = argparse.SUPPRESS,
        **kwargs: object,
    ) -> None:
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: Optional[str] = None,
    ) -> None:
        print("Available presets:\n" + list_presets())
        parser.exit()

def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grade a short-form luxury real-estate video with preset-driven LUT, color science and master deliverable output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_video", type=Path, help="Source video to be mastered.")
    parser.add_argument(
        "output_video", type=Path, help="Destination path for the master grade."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="signature_estate",
        help="Select the baseline grading look to apply.",
    )
    parser.add_argument(
        "--list-presets",
        action=ListPresetsAction,
        help="Print available presets and exit.",
    )
    parser.add_argument(
        "--custom-lut",
        type=Path,
        help="Override the preset LUT with a custom .cube file.",
    )
    parser.add_argument(
        "--lut-strength",
        type=float,
        help="Blend the LUT with the original signal (0.0-1.0).",
    )
    parser.add_argument(
        "--denoise",
        choices=list(HQDN3D_PRESETS) + ["off"],
        help="Override denoise strength.",
    )
    parser.add_argument("--contrast", type=float, help="Override contrast multiplier.")
    parser.add_argument(
        "--saturation", type=float, help="Override saturation multiplier."
    )
    parser.add_argument("--gamma", type=float, help="Override gamma adjustment.")
    parser.add_argument("--brightness", type=float, help="Override brightness offset.")
    parser.add_argument(
        "--warmth", type=float, help="Override warm mid-tone tint (red channel)."
    )
    parser.add_argument(
        "--cool", type=float, help="Override cool mid-tone tint (blue channel)."
    )
    parser.add_argument(
        "--sharpen",
        choices=list(UNSHARP_PRESETS) + ["off"],
        help="Override clarity setting.",
    )
    parser.add_argument("--grain", type=float, help="Override film-grain intensity.")
    parser.add_argument(
        "--tone-map",
        choices=["auto", "off", "hable", "mobius", "reinhard", "bt2390"],
        default="auto",
        help="Tone-mapping operator to apply (auto = detect HDR metadata).",
    )
    parser.add_argument(
        "--tone-map-peak",
        type=float,
        default=1000.0,
        help="Peak nits reference for tone-mapping operators that support it.",
    )
    parser.add_argument(
        "--tone-map-desat",
        type=float,
        default=0.1,
        help="Desaturation factor for tone-mapping (0.0 retains original chroma).",
    )
    parser.add_argument(
        "--deband",
        choices=list(DEBAND_PRESETS) + ["off"],
        default="off",
        help="Apply debanding (gradfun) smoothing after grain.",
    )
    parser.add_argument(
        "--halation-intensity",
        type=float,
        default=0.0,
        help="Strength of halation bloom (0.0 disables).",
    )
    parser.add_argument(
        "--halation-radius",
        type=float,
        default=18.0,
        help="Radius of the halation blur kernel when enabled.",
    )
    parser.add_argument(
        "--halation-threshold",
        type=float,
        help="Luminance threshold (0-1) before highlights feed the halation pass.",
    )
    parser.add_argument(
        "--video-codec", default="prores_ks", help="Video mezzanine codec to use."
    )
    parser.add_argument(
        "--prores-profile",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Apple ProRes profile when using prores_ks (3 = 422 HQ).",
    )
    parser.add_argument("--bitrate", help="Explicit video bitrate target (e.g. 800M).")
    parser.add_argument(
        "--audio-codec",
        default="copy",
        help="Audio codec to use (pcm_s24le for master-grade PCM).",
    )
    parser.add_argument(
        "--audio-bitrate", help="Override audio bitrate when transcoding audio."
    )
    parser.add_argument("--threads", type=int, help="Limit ffmpeg worker threads.")
    parser.add_argument(
        "--preview-frames",
        type=int,
        help="Render only the first N frames for a quick preview.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the ffmpeg command without executing it.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="ffmpeg log level (quiet, warning, info, verbose).",
    )
    parser.add_argument(
        "--target-fps",
        help="Conform the output to this frame rate (e.g. 23.976 or 24000/1001).",
    )
    parser.add_argument(
        "--frame-tolerance",
        type=float,
        default=0.05,
        help="Tolerance in fps when assessing frame-rate drift before conforming.",
    )
    parser.add_argument(
        "--vsync",
        choices=["auto", "cfr", "vfr", "drop", "passthrough"],
        default="cfr",
        help="Video sync method (muxer vsync behavior).",
    )
    parser.add_argument(
        "--print-filter-graph",
        action="store_true",
        help="Print the filter graph in human-readable format for debugging.",
    )
    parser.add_argument(
        "--color-primaries",
        help="Set explicit color primaries (e.g., bt709, bt2020).",
    )
    parser.add_argument(
        "--color-transfer",
        help="Set explicit color transfer characteristics (e.g., bt709, smpte2084).",
    )
    parser.add_argument(
        "--color-space",
        help="Set explicit color space (e.g., bt709, bt2020nc).",
    )
    parser.add_argument(
        "--color-from-source",
        action="store_true",
        help="Copy color metadata from source instead of overriding.",
    )

    return parser.parse_args(list(argv) if argv is not None else None)

def build_config(
    args: argparse.Namespace,
    *,
    target_fps: Optional[str] = None,
    tone_map_plan: Optional[ToneMapPlan] = None,
) -> Dict[str, object]:
    preset = PRESETS[args.preset]
    config = preset.to_dict()

    if args.custom_lut:
        config["lut"] = args.custom_lut
    if args.lut_strength is not None:
        config["lut_strength"] = args.lut_strength
    if args.denoise is not None:
        config["denoise"] = args.denoise
    if args.contrast is not None:
        config["contrast"] = args.contrast
    if args.saturation is not None:
        config["saturation"] = args.saturation
    if args.gamma is not None:
        config["gamma"] = args.gamma
    if args.brightness is not None:
        config["brightness"] = args.brightness
    if args.warmth is not None:
        config["warmth"] = args.warmth
    if args.cool is not None:
        config["cool"] = args.cool
    if args.sharpen is not None:
        config["sharpen"] = args.sharpen
    if args.grain is not None:
        config["grain"] = args.grain
    if args.deband is not None:
        config["deband"] = args.deband
    if args.halation_intensity is not None:
        config["halation_intensity"] = args.halation_intensity
    if args.halation_radius is not None:
        config["halation_radius"] = args.halation_radius
    if args.halation_threshold is not None:
        config["halation_threshold"] = args.halation_threshold
    if target_fps is not None:
        config["target_fps"] = target_fps
    if tone_map_plan and tone_map_plan.enabled:
        config.update(tone_map_plan.config)

    return config

def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)

    ensure_tools_available()

    input_video = args.input_video.expanduser().resolve()
    output_video = args.output_video.expanduser().resolve()
    if not input_video.exists():
        print(f"Input video not found: {input_video}", file=sys.stderr)
        return 2

    if output_video.exists() and not args.overwrite:
        print(
            f"Output file already exists: {output_video}. Use --overwrite to replace.",
            file=sys.stderr,
        )
        return 3

    try:
        probe = probe_source(input_video)
    except RuntimeError as exc:
        print(f"ffprobe error: {exc}", file=sys.stderr)
        return 4

    print("Source clip:", summarize_probe(probe))

    try:
        frame_plan = assess_frame_rate(probe, args.target_fps, args.frame_tolerance)
    except ValueError as exc:
        print(f"Frame-rate configuration error: {exc}", file=sys.stderr)
        return 6

    print(frame_plan.note)

    tone_plan = plan_tone_mapping(args, probe)
    print(tone_plan.note)

    config = build_config(args, target_fps=frame_plan.target, tone_map_plan=tone_plan)

    # Validate grading parameters early with clear error messages
    try:
        contrast = float(config.get("contrast", 1.0))
        if contrast <= 0:
            raise ValueError("contrast must be greater than 0")

        saturation = float(config.get("saturation", 1.0))
        if saturation <= 0:
            raise ValueError("saturation must be greater than 0")

        gamma = float(config.get("gamma", 1.0))
        if gamma <= 0:
            raise ValueError("gamma must be greater than 0")

        brightness = float(config.get("brightness", 0.0))
        if brightness < -1.0 or brightness > 1.0:
            raise ValueError("brightness must be in range [-1.0, 1.0]")
        config["brightness"] = clamp(brightness, -1.0, 1.0)

        lut_strength = float(config.get("lut_strength", 1.0))
        if not (0.0 <= lut_strength <= 1.0):
            raise ValueError("lut_strength must be in range [0.0, 1.0]")

        tone_map_peak = config.get("tone_map_peak")
        if tone_map_peak is not None and float(cast(Any, tone_map_peak)) <= 0.0:
            raise ValueError("tone_map_peak must be greater than 0")

        tone_map_desat = config.get("tone_map_desat")
        if tone_map_desat is not None and not (0.0 <= float(cast(Any, tone_map_desat)) <= 1.0):
            raise ValueError("tone_map_desat must be within [0.0, 1.0]")

        halation_intensity = float(config.get("halation_intensity", 0.0))
        if halation_intensity < 0.0:
            raise ValueError("halation_intensity must be non-negative")
        config["halation_intensity"] = clamp(halation_intensity, 0.0, 1.0)

        halation_radius = float(config.get("halation_radius", 0.0))
        if halation_radius < 0.0:
            raise ValueError("halation_radius must be non-negative")
        config["halation_radius"] = clamp(halation_radius, 0.0, 128.0)

        threshold_value = float(config.get("halation_threshold", 0.6))
        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError("halation_threshold must be within [0.0, 1.0]")
        config["halation_threshold"] = clamp(threshold_value, 0.0, 1.0)

    except ValueError as exc:
        print(f"Parameter validation error: {exc}", file=sys.stderr)
        return 7

    try:
        filter_graph, filter_output = build_filter_graph(config)
    except Exception as exc:  # surface graph errors for clear diagnosis
        print(f"Failed to build filter graph: {exc}", file=sys.stderr)
        return 5

    if args.print_filter_graph:
        print("\nFilter graph (human-readable):")
        filter_nodes = filter_graph.split(";")
        for i, node in enumerate(filter_nodes):
            print(f"  {i+1:2d}. {node}")
        print()

    # Determine color metadata (explicit > source copy > tone-map defaults)
    color_primaries, color_transfer, color_space = determine_color_metadata(args, probe)
    if (
        color_primaries is None
        and color_transfer is None
        and color_space is None
        and tone_plan.enabled
        and tone_plan.metadata != (None, None, None)
    ):
        color_primaries, color_transfer, color_space = tone_plan.metadata

    cmd = build_command(
        input_video,
        output_video,
        filter_graph,
        filter_output,
        overwrite=args.overwrite,
        video_codec=args.video_codec,
        prores_profile=args.prores_profile,
        bitrate=args.bitrate,
        audio_codec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
        threads=args.threads,
        log_level=args.log_level,
        preview_frames=args.preview_frames,
        vsync=args.vsync,
        color_primaries=color_primaries,
        color_transfer=color_transfer,
        color_space=color_space,
    )

    print("\nFFmpeg command:")
    print(" ".join(shlex.quote(part) for part in cmd))

    if args.dry_run:
        print("Dry run requested; exiting before execution.")
        return 0

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print(f"Master grade created at {output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
