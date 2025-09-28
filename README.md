# 800 Picacho Lane — Professional LUT Collection

## Overview
A cutting-edge collection of **16 professional color grading LUTs** featuring innovative **Material Response** technology.

## Table of Contents
- [Collection Contents](#collection-contents)
- [Innovation](#innovation)
- [Usage](#usage)
- [Luxury TIFF Batch Processor](#luxury-tiff-batch-processor)
  - [Key Features](#key-features)
  - [Requirements](#requirements)
  - [Example](#example)
- [Luxury Video Master Grader](#luxury-video-master-grader)
  - [Key Features](#key-features-1)
  - [Requirements](#requirements-1)
  - [Examples](#examples)
- [HDR Production Pipeline](#hdr-production-pipeline)
  - [Key Features](#key-features-2)
  - [Highlights](#highlights)
  - [Requirements](#requirements-2)
  - [Example](#example-1)
- [License](#license)

## Collection Contents
- **Film Emulation**: Kodak 2393, FilmConvert Nitrate  
- **Location Aesthetic**: Montecito Golden Hour, Spanish Colonial Warm  
- **Material Response**: Revolutionary physics-based surface enhancement

## Innovation
**Material Response LUTs** analyze and enhance how different surfaces interact with light—shifting from purely global color transforms to surface-aware rendering that respects highlights, midtones, and micro-contrast differently across materials. The approach keeps natural roll-off in highlights, protects nuanced midtone skin detail, and intelligently refines specular textures so metal, stone, glass, and fabric all retain their distinctive depth.

## Usage
1. Import into DaVinci Resolve, Premiere Pro, or other color-grading software.
2. Apply at **60–80% opacity** initially.
3. Stack multiple LUTs for complex material interactions.

## Luxury TIFF Batch Processor
The repository now includes `luxury_tiff_batch_processor.py`, a high-end batch workflow for polishing large-format TIFF photography prior to digital launch. The script preserves metadata, honors 16-bit source files when [`tifffile`](https://pypi.org/project/tifffile/) is available, and layers tonal, chroma, clarity, and diffusion refinements tuned for ultra-luxury real-estate storytelling.

### Key Features
- Automatically reads and writes IPTC/XMP metadata so campaign details remain intact across exports.
- Maintains 16-bit precision whenever the optional `tifffile` dependency is installed and falls back to Pillow for 8-bit output.
- Offers presets that mirror the LUT families (Signature, Golden Hour, Heritage, etc.) for rapid client alignment.
- Supports per-run overrides for exposure, midtone contrast, vibrance, clarity, glow, and more to accommodate creative direction.
- Provides non-destructive previews with `--dry-run` and mirrors directory trees with `--recursive` for large productions.

### Requirements
- Python 3.11+
- `pip install numpy pillow` (add `tifffile` for lossless 16-bit output)

> **Note:** Earlier revisions triggered `F821` undefined-name lint errors. Pull the latest main branch (or reinstall from the freshest ZIP) to ensure you have the corrected helper that resolves the NumPy dtype handling.

### Example
```bash
python luxury_tiff_batch_processor.py /path/to/raw_tiffs /path/to/output \
  --preset signature --resize-long-edge 7000 --overwrite
```

Override any preset control (exposure, midtone contrast, vibrance, clarity, glow, etc.)
by providing the corresponding command-line flag. Use `--dry-run` to audit the processing
plan without writing files, and `--recursive` to mirror nested shoot-day folders.

## Luxury Video Master Grader

`luxury_video_master_grader.py` brings the same curated aesthetic to short-form motion content. It wraps FFmpeg with preset-driven LUT application, tasteful denoising, clarity and film-grain treatments, then exports a mezzanine-ready Apple ProRes master by default. The pipeline now auto-detects HDR transfers and tone-maps them into a refined BT.709 space, optionally adds ultra-fine debanding and cinematic halation bloom, and keeps gradient-rich interiors spotless with updated presets.

### Key Features
- Intelligent source analysis that detects resolution, frame rate, HDR transfer characteristics, and audio layout before rendering.
- Preset-driven workflows that blend LUT application with spatial/temporal filtering tailored to luxury real-estate cinematography.
- Advanced finishing controls for debanding, halation, clarity, grain, and tone mapping to deliver cinematic masters out of the box.
- Flexible output targets including mezzanine-grade ProRes profiles and user-defined frame rates via `--target-fps`.
- Dry-run inspection that prints the underlying FFmpeg command for validation before committing to long renders.

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

Use `--custom-lut` to feed bespoke `.cube` files, tweak parameters such as `--contrast` or `--grain`, layer in `--deband` smoothing or halation controls, and enable `--dry-run` to inspect the underlying FFmpeg command without rendering. The script automatically probes the source to surface resolution, frame-rate metadata and audio configuration before processing, then monitors for drift or variable frame-rate clips. When necessary it conforms delivery to the nearest cinema broadcast standard (or a user-specified `--target-fps`) to guarantee smooth, continuous playback. HDR clips are further analyzed so the tool can apply tasteful tone mapping automatically, while explicit `--tone-map` overrides give you authoritative control whenever a specific operator is required.

## HDR Production Pipeline

`hdr_production_pipeline.sh` orchestrates a full HDR finishing pass, combining ACES tone mapping, adaptive debanding, and filmic halation for gallery-ready masters. The workflow harmonizes the bespoke Codex automation steps with the broader pipeline overview introduced on main so teams can reference a single, unified set of HDR finishing instructions.

### Key Features
- ACES Output Device Transform (ODT) selection, Dolby Vision metadata pass-through, and HDR10 mastering options for broadcast compliance.
- Adaptive debanding tuned to Codex reference recipes to protect smooth gradients in modern architectural interiors.
- Filmic halation and bloom controls that layer naturally over the LUT aesthetic without introducing color drift.
- Optional tone-mapping operator selection (`--tone-map`) for precise control over HDR-to-SDR conversions.
- Designed to slot after `luxury_video_master_grader.py` so teams can reuse LUT-driven looks while finishing in HDR.

### Highlights
- Consolidates automation steps from the Codex finishing toolkit into a single, reproducible command sequence.
- Documents the hand-off from the LUT-driven grade to HDR-specific finishing so teams can collaborate without guesswork.
- Provides defaults that mirror the examples in this README, making it simple to reproduce the reference masters.

### Requirements
- macOS or Linux shell environment
- FFmpeg with zimg, `ffprobe`, and `python3`

### Example
```bash
./hdr_production_pipeline.sh source_hdr.mov out_hdr_master.mov \
  --aces-odt rec2020-pq --deband medium --halation strong --hdr10-metadata auto
```

Layer the script after `luxury_video_master_grader.py` to apply bespoke LUTs before the
HDR-specific finishing tools run. The pipeline preserves Dolby Vision and static HDR10
metadata where available, while the deband and halation stages default to the Codex branch
recipes highlighted in the documentation examples.

## License
Professional use permitted with attribution.
