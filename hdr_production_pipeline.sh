# file: hdr_pipeline.py
"""
HDR Production Pipeline (Golden Hour)

Features
- Interactive or --batch DIR (per-file subfolders); --recursive; --ext pattern
- --jobs N parallel batch with progress bar (tqdm if available)
- --gpu auto|nvenc|qsv|amf|videotoolbox|x264|x265
- --config YAML to tune CRF/BPP and per-backend extra flags (deep-merged)
- Auto bitrate/CRF from resolution & fps (ffprobe)
- Robust filtergraph escaping for LUT paths
- Color tags, hvc1, +faststart for web outputs
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # fallback UI


# ------------------------- logging --------------------------------------------

def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(message)s")


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
    return " ".join(shlex.quote(a) for a in args)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def escape_filter_value(text: str) -> str:
    # Escape filtergraph-special chars (not shell quoting).
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


# ------------------------- YAML config ----------------------------------------

DEFAULT_CONFIG: Dict[str, object] = {
    "general": {
        "fps_boost_den": 150.0,
        "ffmpeg_log_level": "error",
    },
    "x265": {
        "preset": "slow",
        "hdr10_params": "hdr10=1:hdr10-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
        "crf": {
            "HDR_MASTER": {"2160p": 14, "1440p": 13, "1080p": 12, "720p": 12, "sd": 12},
            "YOUTUBE_HDR": {"2160p": 18, "1440p": 18, "1080p": 19, "720p": 20, "sd": 20},
        },
    },
    "x264": {
        "preset": "slow",
        "crf": {"2160p": 18, "1440p": 18, "1080p": 19, "720p": 21, "sd": 22},
    },
    "hw": {
        "bpp": {"HDR_MASTER": 0.095, "YOUTUBE_HDR": 0.060, "SDR": 0.050},
        "min_bps": {"HDR_MASTER": 24_000_000, "YOUTUBE_HDR": 10_000_000, "SDR": 6_000_000},
        "max_bps": {"HDR_MASTER": 120_000_000, "YOUTUBE_HDR": 60_000_000, "SDR": 35_000_000},
    },
    "backend_extra": {
        "hevc_nvenc": {"HDR_MASTER": ["-profile:v", "main10"], "YOUTUBE_HDR": ["-profile:v", "main10"], "SDR": []},
        "hevc_qsv":   {"HDR_MASTER": ["-profile:v", "main10"], "YOUTUBE_HDR": ["-profile:v", "main10"], "SDR": []},
        "hevc_amf":   {"HDR_MASTER": ["-profile:v", "main10"], "YOUTUBE_HDR": ["-profile:v", "main10"], "SDR": []},
        "hevc_videotoolbox": {"HDR_MASTER": ["-profile:v", "main10"], "YOUTUBE_HDR": ["-profile:v", "main10"], "SDR": []},
        "h264_videotoolbox": {"SDR": []},
        "h264_nvenc": {"SDR": []},
        "h264_qsv":   {"SDR": []},
        "h264_amf":   {"SDR": []},
    },
}

def deep_update(base: MutableMapping[str, object], patch: Mapping[str, object]) -> MutableMapping[str, object]:
    for k, v in patch.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v  # type: ignore[index]
    return base


def load_config(path: Optional[Path]) -> Dict[str, object]:
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    if not path:
        return cfg
    if not path.is_file():
        raise SystemExit(f"error: config not found: {path}")
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(f"error: pyyaml is required to use --config ({e})")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise SystemExit("error: YAML root must be a mapping")
    deep_update(cfg, data)  # type: ignore[arg-type]
    return cfg


# ------------------------- encoders -------------------------------------------

@dataclass
class Encoders:
    hdr: str
    sdr: str
    yt: str  # YouTube HDR encoder


def choose_encoders(pref: str) -> Encoders:
    pref = pref.lower()
    hdr, sdr = "", ""
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
        hdr, sdr = "libx265", "libx264"
    elif pref == "x264":
        hdr, sdr = "libx265", "libx264"
    else:
        raise SystemExit(f"error: unknown --gpu {pref}")
    yt = "libx265" if has_encoder("libx265") else hdr
    return Encoders(hdr=hdr, sdr=sdr, yt=yt)


# ------------------------- probing & heuristics -------------------------------

def ffprobe_value(in_path: Path, entry: str) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={entry}", "-of", "csv=p=0", str(in_path)],
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
    mb = max(1, int(round(bps / 1_000_000.0)))
    return f"{mb}M"


def auto_bitrate(cfg: Mapping[str, object], purpose: str, w: int, h: int, fps: float) -> str:
    hw = cfg["hw"]  # type: ignore[index]
    bpp = float(hw["bpp"][purpose])  # type: ignore[index]
    min_bps = int(hw["min_bps"][purpose])  # type: ignore[index]
    max_bps = int(hw["max_bps"][purpose])  # type: ignore[index]
    bps = int(round(w * h * fps * bpp))
    if fps > 30:
        den = float(cfg["general"]["fps_boost_den"])  # type: ignore[index]
        bps = int(round(bps * (1.0 + ((fps - 30.0) / den))))
    bps = clamp(bps, min_bps, max_bps)
    return human_mbit(bps)


def auto_crf_x265(cfg: Mapping[str, object], purpose: str, rcls: str) -> int:
    table = cfg["x265"]["crf"][purpose]  # type: ignore[index]
    return int(table.get(rcls, table.get("sd", 18)))  # type: ignore[call-arg]


def auto_crf_x264(cfg: Mapping[str, object], rcls: str) -> int:
    table = cfg["x264"]["crf"]  # type: ignore[index]
    return int(table.get(rcls, table.get("sd", 22)))  # type: ignore[call-arg]


def backend_extra(cfg: Mapping[str, object], backend: str, purpose: str) -> List[str]:
    be = cfg.get("backend_extra", {})  # type: ignore[assignment]
    m = be.get(backend, {})  # type: ignore[assignment]
    lst = m.get(purpose, [])  # type: ignore[assignment]
    return list(lst)


def ffmpeg_log_level(cfg: Mapping[str, object]) -> str:
    return str(cfg["general"]["ffmpeg_log_level"])  # type: ignore[index]


# ------------------------- encodes --------------------------------------------

def encode_hdr_master(
    cfg: Mapping[str, object],
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
    cmd: List[str] = ["ffmpeg", "-hide_banner", "-stats", "-loglevel", ffmpeg_log_level(cfg), "-i", str(input_path), "-vf", vf, "-map", "0:v:0", "-map", "0:a?"]
    if enc.hdr == "libx265":
        crf = auto_crf_x265(cfg, "HDR_MASTER", rcls)
        preset = str(cfg["x265"]["preset"])  # type: ignore[index]
        params = str(cfg["x265"]["hdr10_params"])  # type: ignore[index]
        cmd += ["-c:v", "libx265", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p10le", "-x265-params", params, "-tag:v", "hvc1"]
    else:
        br = auto_bitrate(cfg, "HDR_MASTER", w, h, fps)
        cmd += ["-c:v", enc.hdr, "-b:v", br, "-tag:v", "hvc1"]
        cmd += backend_extra(cfg, enc.hdr, "HDR_MASTER")
    cmd += ["-color_primaries", "9", "-color_trc", "16", "-colorspace", "9", "-c:a", "copy", str(out_mov)]
    logging.info("📊 Phase 1: Creating HDR Master… [%dx%d @ %.2ffps, %s, %s]", w, h, fps, enc.hdr, rcls)
    run(cmd, dry_run=dry_run)


def encode_youtube_hdr(
    cfg: Mapping[str, object],
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
    cmd: List[str] = ["ffmpeg", "-hide_banner", "-stats", "-loglevel", ffmpeg_log_level(cfg), "-i", str(hdr_master), "-map", "0:v:0", "-map", "0:a?"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
    if enc.yt == "libx265":
        crf = auto_crf_x265(cfg, "YOUTUBE_HDR", rcls)
        preset = str(cfg["x265"]["preset"])  # type: ignore[index]
        params = str(cfg["x265"]["hdr10_params"])  # type: ignore[index]
        cmd += ["-c:v", "libx265", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p10le", "-x265-params", params, "-tag:v", "hvc1"]
    else:
        br = auto_bitrate(cfg, "YOUTUBE_HDR", base_w, base_h, base_fps)
        cmd += ["-c:v", enc.yt, "-b:v", br, "-tag:v", "hvc1"]
        cmd += backend_extra(cfg, enc.yt, "YOUTUBE_HDR")
    cmd += [str(out_mp4)]
    logging.info("🌐 Phase 2: Creating Web (YouTube HDR)… [%s, %s]", enc.yt, rcls)
    run(cmd, dry_run=dry_run)


def encode_mls_premium_sdr(
    cfg: Mapping[str, object],
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
    cmd: List[str] = ["ffmpeg", "-hide_banner", "-stats", "-loglevel", ffmpeg_log_level(cfg), "-i", str(hdr_master), "-vf", vf, "-map", "0:v:0", "-map", "0:a?"]
    cmd += ["-movflags", "+faststart", "-color_primaries", "1", "-color_trc", "1", "-colorspace", "1"]
    if enc.sdr == "libx264":
        crf = auto_crf_x264(cfg, rcls)
        preset = str(cfg["x264"]["preset"])  # type: ignore[index]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
    else:
        br = auto_bitrate(cfg, "SDR", base_w, base_h, base_fps)
        cmd += ["-c:v", enc.sdr, "-b:v", br]
        cmd += backend_extra(cfg, enc.sdr, "SDR")
    cmd += ["-c:a", "aac", "-b:a", "160k", str(out_mp4)]
    logging.info("🖼️  Phase 3: Creating MLS Premium SDR… [%s, %s]", enc.sdr, rcls)
    run(cmd, dry_run=dry_run)


# ------------------------- discovery & interactive ----------------------------

def iter_files(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    exts = {("." + x.lower().lstrip(".")) for x in re.split(r"\|", pattern, flags=re.IGNORECASE) if x}
    it = root.rglob("*") if recursive else root.glob("*")
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


# ------------------------- per-file processing --------------------------------

def process_one(cfg: Mapping[str, object], input_path: Path, lut_path: Path, out_root: Path, enc: Encoders, *, dry_run: bool = False) -> Path:
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

    encode_hdr_master(cfg, input_path, hdr_master, lut_path, enc, dry_run=dry_run)
    encode_youtube_hdr(cfg, hdr_master, yt_hdr, enc, base_w=w, base_h=h, base_fps=fps, dry_run=dry_run)
    encode_mls_premium_sdr(cfg, hdr_master, mls_prem, enc, base_w=w, base_h=f h, base_fps=fps, dry_run=dry_run)

    logging.info("✨ Done: %s", out_dir)
    return out_dir


# ------------------------- batch orchestration --------------------------------

def collect_batch_files(batch_dir: Path, ext_pattern: str, recursive: bool) -> List[Path]:
    files = list(iter_files(batch_dir, ext_pattern, recursive=recursive))
    files.sort()
    return files


def run_batch(
    cfg: Mapping[str, object],
    files: Sequence[Path],
    lut_path: Path,
    out_root: Path,
    enc: Encoders,
    *,
    jobs: int,
    dry_run: bool,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, str]]]:
    results: List[Tuple[Path, Path]] = []
    errors: List[Tuple[Path, str]] = []
    total = len(files)
    jobs = max(1, int(jobs))

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        fut_map = {ex.submit(process_one, cfg, f, lut_path, out_root, enc, dry_run=dry_run): f for f in files}

        if tqdm is not None:
            with tqdm(total=total, desc="Batch", unit="file") as bar:  # type: ignore[call-arg]
                for fut in as_completed(fut_map):
                    f = fut_map[fut]
                    try:
                        out_dir = fut.result()
                        results.append((f, out_dir))
                    except subprocess.CalledProcessError as e:
                        errors.append((f, f"ffmpeg exit {e.returncode}"))
                    except Exception as e:
                        errors.append((f, str(e)))
                    finally:
                        bar.update(1)
        else:
            done = 0
            for fut in as_completed(fut_map):
                f = fut_map[fut]
                try:
                    out_dir = fut.result()
                    results.append((f, out_dir))
                except subprocess.CalledProcessError as e:
                    errors.append((f, f"ffmpeg exit {e.returncode}"))
                except Exception as e:
                    errors.append((f, str(e)))
                done += 1
                print(f"[{done}/{total}] {f}")

    return results, errors


# ------------------------- cli -------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="HDR Production Pipeline (Golden Hour)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch", type=Path, help="Process all videos in DIR (no prompts).")
    parser.add_argument("--lut", type=Path, help="3D LUT (.cube) applied to inputs (required in --batch).")
    parser.add_argument("--recursive", "-R", action="store_true", help="Recurse in --batch mode.")
    parser.add_argument("--ext", default="mp4|mov", help="Extensions list like 'mp4|mov|mxf'.")
    parser.add_argument("--gpu", default="auto",
                        choices=["auto", "nvenc", "qsv", "amf", "videotoolbox", "x264", "x265"],
                        help="Select encoder family or auto-detect.")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Parallel jobs for --batch.")
    parser.add_argument("--config", type=Path, help="YAML config to tune CRF/BPP and backend flags.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned ffmpeg commands without running.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING).")

    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    require_cmd("ffmpeg")
    require_cmd("ffprobe")
    cfg = load_config(args.config)
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

        logging.info("🗂️  Batch start: %d files → %s (GPU=%s, jobs=%s)", len(files), out_root, args.gpu, args.jobs)
        results, errors = run_batch(cfg, files, args.lut, out_root, enc, jobs=args.jobs, dry_run=args.dry_run)
        logging.info("✅ Batch complete: %s", out_root)
        logging.info("   Succeeded: %d  |  Failed: %d", len(results), len(errors))
        if errors:
            logging.info("   Failures:")
            for f, msg in errors:
                logging.info("   - %s :: %s", f, msg)
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
        out_dir = process_one(cfg, in_path, lut_path, Path.cwd(), enc, dry_run=args.dry_run)
        print(f"Outputs → {out_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\naborted.")