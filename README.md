# 800 Picacho Lane Professional LUT Collection

## Overview
A cutting-edge collection of 16 professional color grading LUTs featuring innovative Material Response technology.

## Collection Contents
- **Film Emulation**: Kodak 2393, FilmConvert Nitrate
- **Location Aesthetic**: Montecito Golden Hour, Spanish Colonial Warm
- **Material Response**: Revolutionary physics-based surface enhancement

## Innovation
Material Response LUTs analyze and enhance how different surfaces interact with light, representing a paradigm shift from traditional color grading.

## Usage
1. Import into DaVinci Resolve, Premiere Pro, or other color grading software
2. Apply at 60-80% opacity initially
3. Stack multiple LUTs for complex material interactions

## Luxury TIFF Batch Processor
The repository now includes `luxury_tiff_batch_processor.py`, a high-end batch workflow
for polishing large-format TIFF photography prior to digital launch. The script preserves
metadata, honours 16-bit source files when [`tifffile`](https://pypi.org/project/tifffile/)
is available, and layers tonal, chroma, clarity, and diffusion refinements tuned for
ultra-luxury real-estate storytelling.

### Requirements
- Python 3.11+
- `pip install numpy pillow` (add `tifffile` for lossless 16-bit output)

### Example
```bash
python luxury_tiff_batch_processor.py /path/to/raw_tiffs /path/to/output \
  --preset signature --resize-long-edge 7000 --overwrite
```

Override any preset control (exposure, midtone contrast, vibrance, clarity, glow, etc.)
by providing the corresponding command-line flag. Use `--dry-run` to audit the processing
plan without writing files, and `--recursive` to mirror nested shoot-day folders.

## Luxury Video Master Grader

`luxury_video_master_grader.py` brings the same curated aesthetic to short-form motion
content. It wraps FFmpeg with preset-driven LUT application, tasteful denoising, clarity
and film-grain treatments, then exports a mezzanine-ready Apple ProRes master by default.
The pipeline now auto-detects HDR transfers and tone-maps them into a refined BT.709
space, optionally adds ultra-fine debanding and cinematic halation bloom, and keeps
gradient-rich interiors spotless with updated presets.

### Requirements
- FFmpeg 6+

### Examples
```bash
# Inspect available looks and recipes
python luxury_video_master_grader.py --list-presets

# Grade a clip with the signature exterior look and generate a ProRes 422 HQ master
python luxury_video_master_grader.py foyer.mov foyer_master.mov --preset signature_estate --overwrite

# Apply the courtyard sunset preset, but intensify saturation and render a 240-frame preview
python luxury_video_master_grader.py pool.mp4 pool_preview.mov \
  --preset golden_hour_courtyard --saturation 1.2 --preview-frames 240 --dry-run

# Force a master at 23.976fps if the source is variable frame rate
python luxury_video_master_grader.py drone.mov drone_master.mov \
  --target-fps 23.976 --overwrite

# Manually invoke the advanced finishing pipeline on an HDR master
python luxury_video_master_grader.py hdr.mov hdr_sdr_master.mov \
  --tone-map hable --tone-map-peak 1200 --tone-map-desat 0.2 \
  --deband strong --halation-intensity 0.18 --halation-radius 22
```

Use `--custom-lut` to feed bespoke `.cube` files, tweak parameters such as `--contrast`
or `--grain`, layer in `--deband` smoothing or halation controls, and enable `--dry-run`
to inspect the underlying FFmpeg command without
rendering. The script automatically probes the source to surface resolution, frame-rate
metadata and audio configuration before processing, then monitors for drift or variable
frame-rate clips. When necessary it conforms delivery to the nearest cinema broadcast
standard (or a user-specified `--target-fps`) to guarantee smooth, continuous playback.
HDR clips are further analysed so the tool can apply tasteful tone mapping automatically,
or respect explicit `--tone-map` overrides when you need a particular operator.

## License
Professional use permitted with attribution.
