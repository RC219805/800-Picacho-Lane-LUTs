"""Luxury video master grading pipeline.

This module orchestrates an FFmpeg-based finishing pass tuned for the
"800 Picacho Lane" look. It applies a curated LUT, contrast and color
balancing refinements, optional spatial denoising, clarity boosts and film
grain, then outputs a mezzanine/master grade file (Apple ProRes by default).

The script mirrors the ergonomics of the TIFF batch processor already in the
repository, exposing a preset-driven command line with opt-in overrides and a
dry-run preview.  A short ffprobe inspection is performed up-front to surface
source metadata before processing.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Repository-rooted LUT presets curated for short-form luxury real-estate video.
REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class GradePreset:
    """Container describing the finishing recipe for a video preset."""

    name: str
    description: str
    lut: Path
    lut_strength: float = 1.0
    denoise: Optional[str] = None
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    brightness: float = 0.0
    warmth: float = 0.0
    cool: float = 0.0
    sharpen: Optional[str] = None
    grain: float = 0.0
    tone_map: Optional[str] = None
    tone_map_peak: Optional[float] = None
    tone_map_desat: Optional[float] = None
    deband: Optional[str] = None
    halation_intensity: float = 0.0
    halation_radius: float = 16.0
    halation_threshold: float = 0.6
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Return a mutable dictionary representation for override merging."""

        data = {
            "lut": self.lut,
            "lut_strength": self.lut_strength,
            "denoise": self.denoise,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "gamma": self.gamma,
            "brightness": self.brightness,
            "warmth": self.warmth,
            "cool": self.cool,
            "sharpen": self.sharpen,
            "grain": self.grain,
            "tone_map": self.tone_map,
            "tone_map_peak": self.tone_map_peak,
            "tone_map_desat": self.tone_map_desat,
            "deband": self.deband,
            "halation_intensity": self.halation_intensity,
            "halation_radius": self.halation_radius,
            "halation_threshold": self.halation_threshold,
        }
        return data


PRESETS: Dict[str, GradePreset] = {
    "signature_estate": GradePreset(
        name="Signature Estate",
        description="Flagship Kodak 2393 emulation with gentle highlight roll-off, soft denoise and warm mid-tones.",
        lut=REPO_ROOT / "01_Film_Emulation" / "Kodak" / "Kodak_2393_D55.cube",
        lut_strength=0.85,
        denoise="soft",
        contrast=1.06,
        saturation=1.10,
        gamma=0.98,
        brightness=0.02,
        warmth=0.015,
        cool=-0.010,
        sharpen="medium",
        grain=6.0,
        deband="fine",
        halation_intensity=0.14,
        halation_radius=20.0,
        halation_threshold=0.52,
        notes="Primary hero look for exterior fly-throughs and architectural establishing shots.",
    ),
    "golden_hour_courtyard": GradePreset(
        name="Golden Hour Courtyard",
        description="Sunset warmth inspired by Montecito golden light with richer saturation and restrained grain.",
        lut=REPO_ROOT
        / "02_Location_Aesthetic"
        / "California"
        / "Montecito_Golden_Hour_HDR.cube",
        lut_strength=0.9,
        denoise="soft",
        contrast=1.04,
        saturation=1.14,
        gamma=0.96,
        brightness=0.015,
        warmth=0.025,
        cool=-0.015,
        sharpen="soft",
        grain=4.0,
        halation_intensity=0.1,
        halation_radius=18.0,
        halation_threshold=0.5,
        notes="Use for west-facing terraces, pool decks and garden lifestyle coverage.",
    ),
    "interior_neutral_luxe": GradePreset(
        name="Interior Neutral Luxe",
        description="Clean, neutral interior pass with FilmConvert Nitrate base, elevated clarity and no added grain.",
        lut=REPO_ROOT
        / "01_Film_Emulation"
        / "FilmConvert"
        / "FilmConvert_Nitrate_LuxuryRE.cube",
        lut_strength=0.8,
        denoise="medium",
        contrast=1.03,
        saturation=1.04,
        gamma=1.0,
        brightness=0.01,
        warmth=0.005,
        cool=0.0,
        sharpen="strong",
        grain=0.0,
        deband="fine",
        notes="Ideal for natural light interiors where texture detail and neutrality are paramount.",
    ),
}


@dataclass
class FrameRatePlan:
    """Summary of how the script will handle frame rate conformance."""

    target: Optional[str]
    note: str


@dataclass
class DynamicRangePlan:
    """Summary of tone-mapping decisions for the master grade."""

    tone_map: Optional[str]
    peak: Optional[float]
    desat: Optional[float]
    transfer: Optional[str]
    primaries: Optional[str]
    note: Optional[str] = None

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
    "fine": (0.65, 16),
    "strong": (0.85, 24),
}

ALLOWED_TONEMAP_METHODS = {"clip", "linear", "gamma", "reinhard", "hable", "mobius"}


def list_presets() -> str:
    lines = []
    for key, preset in PRESETS.items():
        lines.append(f"- {key}: {preset.description}")
    return "\n".join(lines)


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


def probe_source(path: Path) -> Dict[str, object]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Unable to parse ffprobe output") from exc


def summarize_probe(data: Dict[str, object]) -> str:
    fmt = data.get("format", {})
    duration = fmt.get("duration")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    pieces = []
    if duration:
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            pieces.append(f"duration {duration}")
        else:
            pieces.append(f"duration {duration_value:.2f}s")
    if video:
        w = video.get("width")
        h = video.get("height")
        fps = describe_frame_rates(
            video.get("avg_frame_rate"), video.get("r_frame_rate")
        )
        codec = video.get("codec_name") or "unknown"
        video_info = f"video {codec}"
        if w and h:
            video_info += f" {w}x{h}"
        video_info += f" @ {fps}"

        pix_fmt = video.get("pix_fmt")
        if pix_fmt:
            video_info += f" {pix_fmt}"

        bits_per_raw_sample = video.get("bits_per_raw_sample")
        if bits_per_raw_sample:
            video_info += f" {bits_per_raw_sample}bit"
        elif pix_fmt:
            if "10" in pix_fmt:
                video_info += " 10bit"
            elif "12" in pix_fmt:
                video_info += " 12bit"
            elif "16" in pix_fmt:
                video_info += " 16bit"

        color_parts = []
        color_primaries = video.get("color_primaries")
        color_trc = video.get("color_trc")
        colorspace = video.get("colorspace")

        if color_primaries and color_primaries != "unknown":
            color_parts.append(f"primaries={color_primaries}")
        if color_trc and color_trc != "unknown":
            color_parts.append(f"trc={color_trc}")
        if colorspace and colorspace != "unknown":
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
    probe: Dict[str, object],
    user_target: Optional[str],
    tolerance: float,
) -> FrameRatePlan:
    """Evaluate the source frame rate and decide whether to conform it."""

    tolerance = max(tolerance, 0.0001)
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        return FrameRatePlan(target=None, note="No video stream detected; skipping frame-rate adjustments.")

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
    assert base is not None  # for type checkers

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


def assess_dynamic_range(
    probe: Dict[str, object],
    user_method: Optional[str],
    user_peak: Optional[float],
    user_desat: Optional[float],
) -> DynamicRangePlan:
    """Decide how (and if) HDR material should be tonemapped to SDR."""

    video = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        return DynamicRangePlan(None, None, None, None, None, note=None)

    method = (user_method or "").strip().lower() or None
    if method and method not in ALLOWED_TONEMAP_METHODS and method != "off":
        raise ValueError(
            f"Unsupported tone-mapping method '{user_method}'. Choose from {sorted(ALLOWED_TONEMAP_METHODS)} or 'off'."
        )

    transfer = str(video.get("color_trc") or "").lower() or None
    primaries = str(video.get("color_primaries") or "").lower() or None
    if transfer in {"", "unknown", "unspecified"}:
        transfer = None
    if primaries in {"", "unknown", "unspecified"}:
        primaries = None

    auto_method: Optional[str] = None
    auto_note: Optional[str] = None
    if not method:
        hdr_transfers = {"smpte2084", "arib-std-b67"}
        hdr_primaries = {"bt2020"}
        if (transfer and transfer in hdr_transfers) or (primaries and primaries in hdr_primaries):
            auto_method = "hable"
            hdr_desc = ", ".join(filter(None, [transfer or None, primaries or None])) or "HDR metadata"
            auto_note = (
                f"Detected {hdr_desc}; applying automatic tone mapping with '{auto_method}' response."
            )
    elif method == "off":
        return DynamicRangePlan(None, None, None, transfer, primaries, note="Tone mapping disabled by user.")

    chosen = method if method and method != "off" else auto_method
    if not chosen:
        return DynamicRangePlan(None, None, None, transfer, primaries, note="Source flagged as SDR; preserving transfer.")

    peak = user_peak if user_peak is not None else 1000.0
    if peak <= 0:
        raise ValueError("tone-map-peak must be positive when tone mapping is enabled")

    desat = user_desat if user_desat is not None else 0.25
    desat = clamp(desat, 0.0, 1.0)

    note = auto_note
    if method and method != "off":
        note = f"Tone map override requested; using '{chosen}' operator."

    return DynamicRangePlan(chosen, peak, desat, transfer, primaries, note=note)


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

    denoise = config.get("denoise")
    if denoise and denoise.lower() != "off":
        expr = HQDN3D_PRESETS.get(denoise.lower())
        if not expr:
            raise ValueError(f"Unsupported denoise preset: {denoise}")
        new_label = next_label()
        nodes.append(f"[{current}]{expr}[{new_label}]")
        current = new_label

    # Convert to planar RGB float for grading operations.
    new_label = next_label()
    nodes.append(f"[{current}]format=gbrpf32le[{new_label}]")
    current = new_label
    pre_grade_label = current

    tone_map = config.get("tone_map")
    if tone_map and str(tone_map).lower() != "off":
        method = str(tone_map).lower()
        peak = float(config.get("tone_map_peak", 1000.0))
        desat = clamp(float(config.get("tone_map_desat", 0.25)), 0.0, 1.0)
        transfer = str(config.get("tone_map_transfer") or "linear").lower()
        primaries = str(config.get("tone_map_primaries") or "bt709").lower()

        # Normalise to linear light before tone mapping, then return to BT.709 space.
        new_label = next_label()
        nodes.append(
            f"[{current}]zscale=transfer=linear:transfer_in={transfer}:"
            f"primaries=bt709:primaries_in={primaries}:range=full[{new_label}]"
        )
        current = new_label

        new_label = next_label()
        tm_expr = f"tonemap={method}:peak={peak:.1f}:desat={desat:.3f}"
        nodes.append(f"[{current}]{tm_expr}[{new_label}]")
        current = new_label

        new_label = next_label()
        nodes.append(
            f"[{current}]zscale=transfer=bt709:transfer_in=linear:primaries=bt709:primaries_in=bt709:range=full[{new_label}]"
        )
        current = new_label

    pre_grade_label = current

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

    if eq_parts:
        new_label = next_label()
        nodes.append(f"[{current}]eq={':'.join(eq_parts)}[{new_label}]")
        current = new_label

    warmth = float(config.get("warmth", 0.0))
    cool = float(config.get("cool", 0.0))
    if not math.isclose(warmth, 0.0, abs_tol=1e-4) or not math.isclose(cool, 0.0, abs_tol=1e-4):
        new_label = next_label()
        # Clamp values to [-0.5, 0.5] to stay within tasteful limits.
        warmth_c = clamp(warmth, -0.5, 0.5)
        cool_c = clamp(cool, -0.5, 0.5)
        nodes.append(
            f"[{current}]colorbalance=rm={warmth_c:.4f}:gm=0.0000:bm={cool_c:.4f}[{new_label}]"
        )
        current = new_label

    pre_grade_label = current

    lut_path: Path = Path(config["lut"]).resolve()
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    new_label = next_label()
    lut_expr = f"lut3d=file={shlex.quote(str(lut_path))}:interp=tetrahedral"
    nodes.append(f"[{current}]{lut_expr}[{new_label}]")
    current = new_label
    graded_label = current

    lut_strength = float(config.get("lut_strength", 1.0))
    if lut_strength < 0.0 or lut_strength > 1.0:
        raise ValueError("lut_strength must be between 0.0 and 1.0")

    if lut_strength < 0.999:
        blend_label = next_label()
        nodes.append(
            f"[{pre_grade_label}][{graded_label}]blend=all_expr='A*(1-{lut_strength:.4f})+B*{lut_strength:.4f}'[{blend_label}]"
        )
        current = blend_label
    else:
        current = graded_label

    deband = config.get("deband")
    if deband and str(deband).lower() != "off":
        preset = DEBAND_PRESETS.get(str(deband).lower())
        if not preset:
            raise ValueError(f"Unsupported deband preset: {deband}")
        strength, radius = preset
        new_label = next_label()
        nodes.append(f"[{current}]format=yuv444p16le[{new_label}]")
        current = new_label

        new_label = next_label()
        nodes.append(f"[{current}]gradfun=strength={strength:.3f}:radius={int(radius)}[{new_label}]")
        current = new_label

        new_label = next_label()
        nodes.append(f"[{current}]format=gbrpf32le[{new_label}]")
        current = new_label

    halation_intensity = float(config.get("halation_intensity", 0.0))
    if halation_intensity > 0.0:
        intensity = clamp(halation_intensity, 0.0, 1.0)
        radius = max(1.0, float(config.get("halation_radius", 16.0)))
        threshold = clamp(float(config.get("halation_threshold", 0.6)), 0.0, 1.0)

        base_label = next_label()
        halo_label = next_label()
        nodes.append(f"[{current}]split=2[{base_label}][{halo_label}]")

        highlight_label = next_label()
        nodes.append(
            f"[{halo_label}]colorlevels=rimin={threshold:.3f}:gimin={threshold:.3f}:bimin={threshold:.3f}[{highlight_label}]"
        )

        blur_label = next_label()
        nodes.append(f"[{highlight_label}]gblur=sigma={radius:.2f}:steps=2[{blur_label}]")

        tint_label = next_label()
        nodes.append(
            f"[{blur_label}]colorbalance=rm={intensity*0.55:.4f}:gm={intensity*0.25:.4f}:bm={-intensity*0.15:.4f}[{tint_label}]"
        )

        blend_label = next_label()
        nodes.append(f"[{base_label}][{tint_label}]blend=all_expr='A+({intensity:.3f}*B)'[{blend_label}]")
        current = blend_label

    sharpen = config.get("sharpen")
    if sharpen and sharpen.lower() != "off":
        expr = UNSHARP_PRESETS.get(sharpen.lower())
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

    target_fps = config.get("target_fps")
    if target_fps:
        new_label = next_label()
        nodes.append(f"[{current}]fps=fps={target_fps}[{new_label}]")
        current = new_label

    nodes.append(f"[{current}]format=yuv422p10le[vout]")
    graph = ";".join(nodes)
    return graph, "vout"


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
    output_fps: Optional[str],
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

    cmd.extend(["-color_primaries", "1", "-color_trc", "1", "-colorspace", "1"])

    cmd.extend(["-c:a", audio_codec])
    if audio_bitrate:
        cmd.extend(["-b:a", audio_bitrate])

    if preview_frames:
        cmd.extend(["-frames:v", str(preview_frames)])

    if threads:
        cmd.extend(["-threads", str(threads)])

    if output_fps:
        cmd.extend(["-r", output_fps])

    cmd.append(str(output_path))
    return cmd


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grade a short-form luxury real-estate video with preset-driven LUT, color science and master deliverable output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_video", type=Path, nargs="?", help="Source video to be mastered.")
    parser.add_argument("output_video", type=Path, nargs="?", help="Destination path for the master grade.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="signature_estate",
        help="Select the baseline grading look to apply.",
    )
    parser.add_argument("--list-presets", action="store_true", help="Print available presets and exit.")
    parser.add_argument("--custom-lut", type=Path, help="Override the preset LUT with a custom .cube file.")
    parser.add_argument("--lut-strength", type=float, help="Blend the LUT with the original signal (0.0-1.0).")
    parser.add_argument("--denoise", choices=list(HQDN3D_PRESETS) + ["off"], help="Override denoise strength.")
    parser.add_argument("--contrast", type=float, help="Override contrast multiplier.")
    parser.add_argument("--saturation", type=float, help="Override saturation multiplier.")
    parser.add_argument("--gamma", type=float, help="Override gamma adjustment.")
    parser.add_argument("--brightness", type=float, help="Override brightness offset.")
    parser.add_argument("--warmth", type=float, help="Override warm mid-tone tint (red channel).")
    parser.add_argument("--cool", type=float, help="Override cool mid-tone tint (blue channel).")
    parser.add_argument("--sharpen", choices=list(UNSHARP_PRESETS) + ["off"], help="Override clarity setting.")
    parser.add_argument("--grain", type=float, help="Override film-grain intensity.")
    parser.add_argument("--video-codec", default="prores_ks", help="Video mezzanine codec to use.")
    parser.add_argument(
        "--prores-profile",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Apple ProRes profile when using prores_ks (3 = 422 HQ).",
    )
    parser.add_argument("--bitrate", help="Explicit video bitrate target (e.g. 800M).")
    parser.add_argument("--audio-codec", default="copy", help="Audio codec to use (pcm_s24le for master-grade PCM).")
    parser.add_argument("--audio-bitrate", help="Override audio bitrate when transcoding audio.")
    parser.add_argument("--threads", type=int, help="Limit ffmpeg worker threads.")
    parser.add_argument("--preview-frames", type=int, help="Render only the first N frames for a quick preview.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Log the ffmpeg command without executing it.")
    parser.add_argument("--log-level", default="info", help="ffmpeg log level (quiet, warning, info, verbose).")
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
        "--tone-map",
        choices=sorted(ALLOWED_TONEMAP_METHODS) + ["off"],
        help="Override tone-mapping operator (defaults to automatic HDR detection).",
    )
    parser.add_argument(
        "--tone-map-peak",
        type=float,
        help="Assumed HDR peak luminance in nits when tone mapping (positive float).",
    )
    parser.add_argument(
        "--tone-map-desat",
        type=float,
        help="Desaturation factor applied during tone mapping (0-1).",
    )
    parser.add_argument(
        "--deband",
        choices=list(DEBAND_PRESETS) + ["off"],
        help="Apply high-quality debanding to polish gradients (preset strength).",
    )
    parser.add_argument(
        "--halation-intensity",
        type=float,
        help="Strength of cinematic halation bloom (set 0 to disable).",
    )
    parser.add_argument(
        "--halation-radius",
        type=float,
        help="Blur radius used when generating the halation glow mask.",
    )
    parser.add_argument(
        "--halation-threshold",
        type=float,
        help="Luminance threshold (0-1) before highlights feed the halation pass.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.list_presets and (args.input_video is None or args.output_video is None):
        parser.error("input_video and output_video are required unless --list-presets is provided")
    return args


def build_config(
    args: argparse.Namespace,
    *,
    target_fps: Optional[str] = None,
    tone_map: Optional[str] = None,
    tone_map_peak: Optional[float] = None,
    tone_map_desat: Optional[float] = None,
    tone_map_transfer: Optional[str] = None,
    tone_map_primaries: Optional[str] = None,
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
    if args.tone_map is not None:
        config["tone_map"] = args.tone_map
    if args.tone_map_peak is not None:
        config["tone_map_peak"] = args.tone_map_peak
    if args.tone_map_desat is not None:
        config["tone_map_desat"] = args.tone_map_desat
    if target_fps is not None:
        config["target_fps"] = target_fps
    if tone_map_transfer is not None:
        config["tone_map_transfer"] = tone_map_transfer
    if tone_map_primaries is not None:
        config["tone_map_primaries"] = tone_map_primaries
    return config


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)

    if args.list_presets:
        print("Available presets:\n" + list_presets())
        return 0

    ensure_tools_available()

    input_video = args.input_video.expanduser().resolve()
    output_video = args.output_video.expanduser().resolve()
    if not input_video.exists():
        print(f"Input video not found: {input_video}", file=sys.stderr)
        return 2

    if output_video.exists() and not args.overwrite:
        print(f"Output file already exists: {output_video}. Use --overwrite to replace.", file=sys.stderr)
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

    try:
        dynamic_plan = assess_dynamic_range(probe, args.tone_map, args.tone_map_peak, args.tone_map_desat)
    except ValueError as exc:
        print(f"Tone-mapping configuration error: {exc}", file=sys.stderr)
        return 7

    if dynamic_plan.note:
        print(dynamic_plan.note)

    config = build_config(
        args,
        target_fps=frame_plan.target,
        tone_map=dynamic_plan.tone_map,
        tone_map_peak=dynamic_plan.peak,
        tone_map_desat=dynamic_plan.desat,
        tone_map_transfer=dynamic_plan.transfer,
        tone_map_primaries=dynamic_plan.primaries,
    )

    try:
        filter_graph, filter_output = build_filter_graph(config)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Failed to build filter graph: {exc}", file=sys.stderr)
        return 5

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
        output_fps=config.get("target_fps"),
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
