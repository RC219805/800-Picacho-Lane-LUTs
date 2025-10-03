# 800 Picacho Lane — Professional LUT Collection

## Overview

A cutting-edge collection of **16 professional color grading LUTs** featuring innovative **Material Response** technology.

## Table of Contents

- [Overview](#overview)
- [Collection Contents](#collection-contents)
- [Innovation](#innovation)
- [Usage](#usage)
- [Luxury TIFF Batch Processor](#luxury-tiff-batch-processor)
    - [TIFF Batch Processor Requirements](#tiff-batch-processor-requirements)
    - [TIFF Batch Processor Example](#tiff-batch-processor-example)
- [Luxury Video Master Grader](#luxury-video-master-grader)
    - [Luxury Video Master Grader Requirements](#luxury-video-master-grader-requirements)
    - [Luxury Video Master Grader Examples](#luxury-video-master-grader-examples)
- [HDR Production Pipeline](#hdr-production-pipeline)
    - [HDR Production Pipeline Highlights](#hdr-production-pipeline-highlights)
    - [HDR Production Pipeline Requirements](#hdr-production-pipeline-requirements)
    - [HDR Production Pipeline Example](#hdr-production-pipeline-example)
- [License](#license)

## Collection Contents

- **Film Emulation**: Kodak 2393, FilmConvert Nitrate
- **Location Aesthetic**: Montecito Golden Hour, Spanish Colonial Warm
- **Material Response**: Revolutionary physics-based surface enhancement

## Innovation

**Material Response LUTs** analyze and enhance how different surfaces interact with light—shifting from purely global color transforms to surface-aware rendering that respects highlights, midtones, and micro-contrast differently across materials.

## Usage

1. Import into DaVinci Resolve, Premiere Pro, or other color-grading software.
2. Apply at **60–80% opacity** initially.
3. Stack multiple LUTs for complex material interactions.

## Luxury TIFF Batch Processor

The repository now includes `luxury_tiff_batch_processor.py`, a high-end batch workflow
for polishing large-format TIFF photography prior to digital launch. The script preserves
metadata, honors 16-bit source files when [`tifffile`](https://pypi.org/project/tifffile/)
is available, and layers tonal, chroma, clarity, and diffusion refinements tuned for
ultra-luxury real-estate storytelling.

### TIFF Batch Processor Requirements

- Python 3.11+
- `pip install numpy pillow` (add `tifffile` for lossless 16-bit output)

> **Note:** Earlier revisions triggered `F821` undefined-name lint errors. Pull the latest
> main branch (or reinstall from the freshest ZIP) to ensure you have the corrected helper
> that resolves the NumPy dtype handling.

### TIFF Batch Processor Example

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
The pipeline now auto-detects HDR transfers and performs tone mapping to convert them into a refined BT.709
space, optionally adds ultra-fine debanding and cinematic halation bloom, and keeps
gradient-rich interiors spotless with updated presets.

### Luxury Video Master Grader Requirements

- FFmpeg 6+

### Luxury Video Master Grader Examples

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
to inspect the underlying FFmpeg command without rendering. The script automatically
probes the source to surface resolution, frame-rate metadata and audio configuration
before processing, then monitors for drift or variable frame-rate clips. When necessary
it conforms delivery to the nearest cinema broadcast standard (or a user-specified
`--target-fps`) to guarantee smooth, continuous playback. HDR clips are further analyzed
so the tool can apply tasteful tone mapping automatically, while explicit
`--tone-map` overrides give you authoritative control whenever a specific operator is
required.

## HDR Production Pipeline

`hdr_production_pipeline.sh` orchestrates a full HDR finishing pass, combining ACES
tone mapping, adaptive debanding, and filmic halation for gallery-ready masters. The
workflow harmonizes the bespoke Codex automation steps with the broader pipeline
overview introduced on main so teams can reference a single, unified set of HDR
finishing instructions.

### HDR Production Pipeline Highlights

- ACES output transform selection with Dolby Vision and HDR10 metadata preservation
- Adaptive debanding tuned to Codex reference recipes
- Filmic halation and finishing touches that complement the luxury master grader

### HDR Production Pipeline Requirements

- macOS or Linux shell environment
- FFmpeg with zimg, `ffprobe`, and `python3`

### HDR Production Pipeline Example

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
