# file: hdr_pipeline.py
"""
800 Picacho Lane - HDR Production Pipeline (Python)

Features:
- Batch mode (--batch) with per-file subfolders, interactive mode otherwise.
- Cross-platform encoder selection: NVENC/QSV/AMF/VideoToolbox/libx26x.
- Auto bitrate/CRF tuning based on resolution and fps via ffprobe.
- Robust escaping for FFmpeg filtergraph values (LUT paths).
- Color metadata, hvc1 tag, faststart for web outputs.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# ------------------------- logging --------------------------------------------

def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(message)s",
    )


# ------------------------- deps & utils ---------------------------------------

def require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"error: missing dependency: {name}")


def has_encoder(name: str) -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "0", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        ).stdout
    except Exception:
        return False
    pattern = rf"(^|\s){re.escape(name)}(\s|$)"
    return re.search(pattern, out, flags=re.MULTILINE) is not None


def shell_join(args: Sequence[str]) -> str:
    # Python 3.8-friendly join for nice preview/dry-run logs.
    return " ".join(shlex.quote(a) for a in args)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def escape_filter_value(text: str) -> str:
    """
    Escape FFmpeg filtergraph special chars in a value (NOT shell quoting).
    Why: prevents filtergraph parsing splits for paths containing these chars.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("=", "\\=")
    text = text.replace(",", "\\,")
    text = text.replace(";", "\\;")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    return text


def run(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    logging.info("→ %s", shell_join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def try_exiftool_gps(in_path: Path) -> Tuple[str, str]:
    if shutil.which("exiftool") is None:
        return "", ""
    # Use explicit arg termination to be safe with leading dashes in filenames.
    def _get(tag: str) -> str:
        try:
            res = subprocess.run(
                ["exiftool", tag, "-s3", "--", str(in_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
            )
            return (res.stdout or "").strip()
        except Exception:
            return ""
    return _get("-GPSLatitude#"), _get("-GPSLongitude#")


# ------------------------- encoders -------------------------------------------

@dataclass
class Encoders:
    hdr: str
    sdr: str
    yt: str  # YouTube HDR encoder (prefer libx265 if present)


def choose_encoders(pref: str) -> Encoders:
    pref = pref.lower()
    hdr = ""
    sdr = ""

    if pref in {"auto", "autodetect"}:
        if sys.platform == "darwin" and has_encoder("hevc_videotoolbox"):
            hdr, sdr = "hevc_videotoolbox", "h264_videotoolbox"
        else:
            if has_encoder("hevc_nvenc"):
                hdr, sdr = "hevc_nvenc", "h264_nvenc"
            elif has_encoder("hevc_qsv"):
                hdr, sdr = "hevc_qsv", "h264_qsv"
            elif has_encoder("hevc_amf"):
                hdr, sdr = "hevc_amf", "h264_amf"
        if not hdr:
            hdr = "libx265"
        if not sdr:
            sdr = "libx264"
    elif pref == "nvenc":
        if not has_encoder("hevc_nvenc"):
            raise SystemExit("error: hevc_nvenc not found")
        hdr, sdr = "hevc_nvenc", "h264_nvenc"
    elif pref == "qsv":
        if not has_encoder("hevc_qsv"):
            raise SystemExit("error: hevc_qsv not found")
        hdr, sdr = "hevc_qsv", "h264_qsv"
    elif pref == "amf":
        if not has_encoder("hevc_amf"):
            raise SystemExit("error: hevc_amf not found")
        hdr, sdr = "hevc_amf", "h264_amf"
    elif pref == "videotoolbox":
        if not has_encoder("hevc_videotoolbox"):
            raise SystemExit("error: hevc_videotoolbox not found")
        hdr, sdr = "hevc_videotoolbox", "h264_videotoolbox"
    elif pref == "x265":
        hdr, sdr = "libx265", "libx264" if has_encoder("libx264") else "libx264"
    elif pref == "x264":
        hdr, sdr = "libx265", "libx264"
    else:
        raise SystemExit(f"error: unknown --gpu {pref}")

    yt = "libx265" if has_encoder("libx265") else hdr
    return Encoders(hdr=hdr, sdr=sdr, yt=yt)


# ------------------------- probing & heuristics -------------------------------

def ffprobe_value(in_path: Path, entry: str) -> str:
    return subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", f"stream={entry}",
            "-of", "csv=p=0",
            str(in_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    ).stdout.strip()


def probe_video(in_path: Path) -> Tuple[int, int, float]:
    w = int(ffprobe_value(in_path, "width") or 0)
    h = int(ffprobe_value(in_path, "height") or 0)
    fps_str = ffprobe_value(in_path, "avg_frame_rate") or "0/0"
    if fps_str in {"0/0", "N/A", ""}:
        fps_str = ffprobe_value(in_path, "r_frame_rate") or "30/1"
    # Avoid float precision pitfalls in shell; but here Python is fine.
    if "/" in fps_str:
        num, den = fps_str.split("/", 1)
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str or 30.0)
    if fps <= 0:
        fps = 30.0
    return w, h, round(fps, 2)


def res_class(w: int, h: int) -> str:
    big = max(w, h)
    if h >= 2160 or big >= 3840:
        return "2160p"
    if h >= 1440 or big >= 2560:
        return "1440p"
    if h >= 1080 or big >= 1920:
        return "1080p"
    if h >= 720 or big >= 1280:
        return "720p"
    return "sd"


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def human_mbit(bps: int) -> str:
    # Mbit/s rounded
    mb = max(1, int(round(bps / 1_000_000.0)))
    return f"{mb}M"


def auto_bitrate(w: int, h: int, fps: float, bpp: float, min_bps: int, max_bps: int) -> str:
    # bps = pixels * fps * bits-per-pixel; boost for high fps
    bps = int(round(w * h * fps * bpp))
    if fps > 30:
        # ~ +20% at 60p, +40% at 120p
        bps = int(round(bps * (1.0 + ((fps - 30.0) / 150.0))))
    bps = clamp(bps, min_bps, max_bps)
    return human_mbit(bps)


def auto_crf_x265(purpose: str, rcls: str) -> int:
    if purpose == "HDR_MASTER":
        return {"2160p": 14, "1440p": 13}.get(rcls, 12)
    # YouTube HDR
    return {"2160p": 18, "1440p": 18, "1080p": 19}.get(rcls, 20)


def auto_crf_x264(rcls: str) -> int:
    return {"2160p": 18, "1440p": 18, "1080p": 19, "720p": 21}.get(rcls, 22)


# ------------------------- encodes --------------------------------------------

FFMPEG_LOG: Sequence[str] = ("-hide_banner", "-stats", "-loglevel", "error")


def encode_hdr_master(
    input_path: Path,
    out_mov: Path,
    lut_path: Path,
    enc: Encoders,
    *,
    dry_run: bool = False,
) -> None:
    w, h, fps = probe_video(input_path)
    rcls = res_class(w, h)
    lut_esc = escape_filter_value(str(lut_path))
    vf = (
        f"format=p010le,"
        f"lut3d=file={lut_esc},"
        f"zscale=t=linear:npl=100,"
        f"zscale=t=smpte2084:p=bt2020:m=bt2020nc:r=tv,"
        f"format=yuv420p10le"
    )
    cmd: List[str] = ["ffmpeg", *FFMPEG_LOG, "-i", str(input_path), "-vf", vf, "-map", "0:v:0", "-map", "0:a?"]
    if enc.hdr == "libx265":
        crf = auto_crf_x265("HDR_MASTER", rcls)
        # Use hdr10 flags; keep 10-bit
        cmd += [
            "-c:v", "libx265", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p10le",
            "-x265-params",
            "hdr10=1:hdr10-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
            "-tag:v", "hvc1",
        ]
    else:
        br = auto_bitrate(w, h, fps, bpp=0.095, min_bps=24_000_000, max_bps=120_000_000)
        cmd += ["-c:v", enc.hdr, "-profile:v", "main10", "-b:v", br, "-tag:v", "hvc1"]

    cmd += [
        "-color_primaries", "9", "-color_trc", "16", "-colorspace", "9",
        "-c:a", "copy", str(out_mov),
    ]

    logging.info("📊 Phase 1: Creating HDR Master… [%dx%d @ %.2ffps, %s, %s]", w, h, fps, enc.hdr, rcls)
    run(cmd, dry_run=dry_run)


def encode_youtube_hdr(
    hdr_master: Path,
    out_mp4: Path,
    enc: Encoders,
    *,
    base_w: int,
    base_h: int,
    base_fps: float,
    dry_run: bool = False,
) -> None:
    rcls = res_class(base_w, base_h)
    cmd: List[str] = ["ffmpeg", *FFMPEG_LOG, "-i", str(hdr_master), "-map", "0:v:0", "-map", "0:a?"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]

    if enc.yt == "libx265":
        crf = auto_crf_x265("YOUTUBE_HDR", rcls)
        cmd += [
            "-c:v", "libx265", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p10le",
            "-x265-params",
            "hdr10=1:hdr10-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
            "-tag:v", "hvc1",
        ]
    else:
        br = auto_bitrate(base_w, base_h, base_fps, bpp=0.060, min_bps=10_000_000, max_bps=60_000_000)
        cmd += ["-c:v", enc.yt, "-b:v", br, "-profile:v", "main10", "-tag:v", "hvc1"]

    cmd += [str(out_mp4)]
    logging.info("🌐 Phase 2: Creating Web (YouTube HDR)… [%s, %s]", enc.yt, rcls)
    run(cmd, dry_run=dry_run)


def encode_mls_premium_sdr(
    hdr_master: Path,
    out_mp4: Path,
    enc: Encoders,
    *,
    base_w: int,
    base_h: int,
    base_fps: float,
    dry_run: bool = False,
) -> None:
    rcls = res_class(base_w, base_h)
    vf = "zscale=t=linear:npl=100,tonemap=hable,zscale=t=bt709:p=bt709:m=bt709,format=yuv420p"
    cmd: List[str] = ["ffmpeg", *FFMPEG_LOG, "-i", str(hdr_master), "-vf", vf, "-map", "0:v:0", "-map", "0:a?"]
    cmd += ["-movflags", "+faststart", "-color_primaries", "1", "-color_trc", "1", "-colorspace", "1"]

    if enc.sdr == "libx264":
        crf = auto_crf_x264(rcls)
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf)]
    else:
        br = auto_bitrate(base_w, base_h, base_fps, bpp=0.050, min_bps=6_000_000, max_bps=35_000_000)
        cmd += ["-c:v", enc.sdr, "-b:v", br]

    cmd += ["-c:a", "aac", "-b:a", "160k", str(out_mp4)]
    logging.info("🖼️  Phase 3: Creating MLS Premium SDR… [%s, %s]", enc.sdr, rcls)
    run(cmd, dry_run=dry_run)


# ------------------------- interactive search ---------------------------------

def iter_files(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    # Simple case-insensitive extension filter based on suffixes list e.g. "mp4|mov".
    exts = {("." + x.lower().lstrip(".")) for x in re.split(r"\|", pattern, flags=re.IGNORECASE) if x}
    if recursive:
        it = root.rglob("*")
    else:
        it = root.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def search_and_select(prompt: str, pattern: str, root: Path = Path(".")) -> Optional[Path]:
    while True:
        try:
            query = input(prompt)
        except EOFError:
            return None
        if not query:
            return None
        matches = [p for p in iter_files(root, pattern, recursive=True) if query.lower() in p.name.lower()]
        if not matches:
            print(f"No matches found for '{query}'.")
            continue
        if len(matches) == 1:
            return matches[0]
        print("Select a file:")
        for idx, p in enumerate(matches, 1):
            print(f"{idx:2d}) {p}")
        try:
            sel = int(input("Enter number: "))
            if 1 <= sel <= len(matches):
                return matches[sel - 1]
        except (ValueError, EOFError):
            pass
        print("Invalid selection.\n")


# ------------------------- processing -----------------------------------------

def process_one(input_path: Path, lut_path: Path, out_root: Path, enc: Encoders, *, dry_run: bool = False) -> Path:
    lat, lon = try_exiftool_gps(input_path)
    coord_suffix = f"_{lat}_{lon}" if lat and lon else ""
    ts = timestamp()
    basename = input_path.stem
    out_dir = out_root / f"HDR_Production_{basename}_{ts}{coord_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    hdr_master = out_dir / f"{basename}_HDR_Master.mov"
    yt_hdr = out_dir / f"{basename}_YouTube_HDR.mp4"
    mls_prem = out_dir / f"{basename}_MLS_Premium.mp4"

    w, h, fps = probe_video(input_path)
    logging.info("▶️  Processing: %s (%dx%d @ %.2ffps)", input_path, w, h, fps)

    encode_hdr_master(input_path, hdr_master, lut_path, enc, dry_run=dry_run)
    # Use probed values from input for rate tuning on deliverables
    encode_youtube_hdr(hdr_master, yt_hdr, enc, base_w=w, base_h=h, base_fps=fps, dry_run=dry_run)
    encode_mls_premium_sdr(hdr_master, mls_prem, enc, base_w=w, base_h=h, base_fps=fps, dry_run=dry_run)

    logging.info("✨ Done: %s", out_dir)
    return out_dir


# ------------------------- cli -------------------------------------------------

def collect_batch_files(batch_dir: Path, ext_pattern: str, recursive: bool) -> List[Path]:
    files = list(iter_files(batch_dir, ext_pattern, recursive=recursive))
    files.sort()
    return files


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="HDR Production Pipeline (Golden Hour)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch", type=Path, help="Process all videos in DIR (no prompts).")
    parser.add_argument("--lut", type=Path, help="3D LUT (.cube) applied to inputs (required in --batch).")
    parser.add_argument("--recursive", "-R", action="store_true", help="Recurse in --batch mode.")
    parser.add_argument("--ext", default="mp4|mov", help="Extensions regex-like list for input discovery.")
    parser.add_argument("--gpu", default="auto",
                        choices=["auto", "nvenc", "qsv", "amf", "videotoolbox", "x264", "x265"],
                        help="Select encoder family or auto-detect.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned ffmpeg commands without running.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING).")

    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    require_cmd("ffmpeg")
    require_cmd("ffprobe")
    enc = choose_encoders(args.gpu)

    if args.batch:
        if not args.lut:
            parser.error("--lut is required in --batch mode")
        if not args.batch.is_dir():
            parser.error(f"--batch dir not found: {args.batch}")
        if not args.lut.is_file():
            parser.error(f"--lut not found: {args.lut}")

        out_root = Path.cwd() / f"HDR_Batch_{timestamp()}"
        out_root.mkdir(parents=True, exist_ok=True)

        files = collect_batch_files(args.batch, args.ext, recursive=args.recursive)
        if not files:
            raise SystemExit(f"error: no input files found in {args.batch} (exts: {args.ext})")

        logging.info("🗂️  Batch start: %d files → %s (GPU=%s)", len(files), out_root, args.gpu)
        for f in files:
            try:
                process_one(f, args.lut, out_root, enc, dry_run=args.dry_run)
            except subprocess.CalledProcessError as e:
                logging.error("failed: %s (exit %s)", f, e.returncode)
        logging.info("✅ Batch complete: %s", out_root)
        return 0

    # Interactive mode
    print(f"Interactive mode (GPU={args.gpu}). Press Enter to exit.")
    while True:
        in_path = search_and_select("Search term for input video (Enter to finish): ", args.ext)
        if not in_path:
            break
        lut_path = search_and_select("Search term for 3D LUT: ", "cube")
        if not lut_path:
            print("No LUT selected; try again.")
            continue
        out_dir = process_one(in_path, lut_path, Path.cwd(), enc, dry_run=args.dry_run)
        print(f"Outputs → {out_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\naborted.")