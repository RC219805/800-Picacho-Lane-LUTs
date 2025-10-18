# path: README.md
[![CI](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/ci.yml/badge.svg)](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Attribution-blue.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/build-passing-success.svg)](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions)

# 800 Picacho Lane — Professional LUT Collection

## Overview
A cutting-edge collection of **16 professional color grading LUTs** featuring innovative **Material Response** technology.

## Quickstart

### Requirements
- Python **3.10+**
- TIFF pipeline: [`tifffile`](https://pypi.org/project/tifffile/) (optional but recommended), `imagecodecs` for LZW/JPEG where available
- FFmpeg (for video tools)

### Install and run your first TIFF enhancement

```bash
# Option A: local install (recommended for this repo)
python -m pip install .

# Option B: from requirements
python -m pip install -r requirements.txt

Process a folder of TIFFs (preferred console script):

lux-batch input_folder --out-dir processed_images --preset signature

Or via the module/CLI script:

python luxury_tiff_batch_processor.py input_folder processed_images --preset signature

For advanced render finishing, see Material Response Finishing for Neural Renders.

Table of Contents
	•	Overview
	•	Quickstart
	•	Collection Contents
	•	Innovation
	•	Usage
	•	Material Response Finishing for Neural Renders
	•	Developer Setup
	•	Install Dependencies
	•	Console Scripts
	•	Test Shortcuts
	•	Luxury TIFF Batch Processor
	•	Luxury Video Master Grader
	•	HDR Production Pipeline
	•	Board Material Aerial Enhancer
	•	Decision Decay Dashboard
	•	v1.3 additions
	•	License

Collection Contents
	•	Film Emulation: Kodak 2393, FilmConvert Nitrate
	•	Location Aesthetic: Montecito Golden Hour, Spanish Colonial Warm
	•	Material Response: Revolutionary physics-based surface enhancement

⸻

Innovation

Material Response LUTs analyze and enhance how different surfaces interact with light—shifting from purely global color transforms to surface-aware rendering that respects highlights, midtones, and micro-contrast differently across materials.

⸻

Usage
	1.	Import into DaVinci Resolve, Premiere Pro, or other color-grading software.
	2.	Apply at 60–80% opacity initially.
	3.	Stack multiple LUTs for complex material interactions.

Material Response Finishing for Neural Renders

lux_render_pipeline.py exposes a Material Response finishing layer that reinforces wood grain, textile separation, fireplace warmth, and atmospheric haze directly from the command line. Enable it with --material-response to activate detail boosts, contact shadowing, and volumetric tinting that better fuse interior renders with their exterior vistas.

Requires: the [ml] extras installed (pip install -e ".[ml]") and a GPU-enabled environment for optimal performance.

Example:

python lux_render_pipeline.py \
  --input bedroom_render.jpg \
  --out ./enhanced_bedroom \
  --prompt "minimalist bedroom interior..." \
  --material-response --texture-boost 0.28


⸻

Developer Setup

Install Dependencies
	•	Create a .venv (optional but recommended)
	•	Install the project requirements:

python -m pip install .
# or mirror CI:
python -m pip install -r requirements.txt

	•	Add optional extras:

pip install -e ".[tiff]"   # 16-bit TIFF processing
pip install -e ".[dev]"    # pytest, linting
pip install -e ".[ml]"     # ML extras for lux_render_pipeline
pip install -e ".[all]"    # everything

Console Scripts

After installation, the following command-line tools are available:
	•	lux-batch — batch process TIFFs (mirrors directory structure, presets)
	•	luxury_tiff_batch_processor.py — batch TIFF processing
	•	luxury_video_master_grader.py — video grading
	•	lux_render_pipeline.py — AI-powered render refinement
	•	decision_decay_dashboard.py — codebase philosophy audits

Test Shortcuts

Use the bundled Makefile:

make test-fast
make test-full
# Tip: `make ci` runs linting (flake8, pylint) and fast tests, mirroring GitHub Actions.


⸻

Luxury TIFF Batch Processor

luxury_tiff_batch_processor.py is a high-end workflow for polishing large-format TIFF photography prior to digital launch. It preserves metadata, honors 16-bit source files (via tifffile), and layers tonal and chroma refinements tuned for luxury real-estate storytelling.

⸻

Luxury Video Master Grader

luxury_video_master_grader.py brings the same curated aesthetic to motion content using FFmpeg.

⸻

HDR Production Pipeline

hdr_production_pipeline.sh orchestrates a full HDR finishing pass, combining ACES tone mapping, adaptive debanding, and halation.

⸻

Board Material Aerial Enhancer

board_material_aerial_enhancer.py applies MBAR-approved material palettes to aerials using clustering and texture blending.

For full documentation, see Palette Assignment Guide.

⸻

Decision Decay Dashboard

decision_decay_dashboard.py surfaces temporal contracts, codebase philosophy violations, and brand color token drift in one terminal dashboard.

⸻

v1.3 additions

Auto-measure eye-line & gutters

python presence_cli_v1_3.py measure \
  --image In-Command_In-Conversation_2400x3000.jpg \
  --aspect 4:5
# -> JSON report printed to stdout

JSON fields

field	type	description
image	string	Path to the analyzed image (as provided).
width	int	Image width in pixels.
height	int	Image height in pixels.
aspect_input	string	Target aspect ratio in W:H form (e.g., 4:5, 1.91:1).
eye_line_pct	number	Estimated eye-line position as percent of image height (0–100, top→bottom).
gutters	object	Suggested letterboxing to reach the target aspect. Integer pixel values per side.
├ left_px	int	Left gutter pixels.
├ right_px	int	Right gutter pixels.
├ top_px	int	Top gutter pixels.
└ bottom_px	int	Bottom gutter pixels.
confidence	number	Heuristic confidence 0.0–1.0. Higher with face detection; reduced for tiny/flat images.
method	string	"face" when OpenCV face is found, otherwise "fallback" (edge-based estimate).

Example output

{
  "image": "In-Command_In-Conversation_2400x3000.jpg",
  "width": 2400,
  "height": 3000,
  "aspect_input": "4:5",
  "eye_line_pct": 41.87,
  "gutters": { "left_px": 0, "right_px": 0, "top_px": 0, "bottom_px": 0 },
  "confidence": 0.88,
  "method": "face"
}

Accuracy notes
	•	Face method (OpenCV Haar cascade) anchors the eye line ~42% from face top—robust for portraits.
	•	Fallback (edge energy, upper field) works without OpenCV; confidence slightly lower.
	•	Very small or low-contrast images reduce confidence.

Optional dependency: Installing OpenCV improves detection:

pip install opencv-python



⸻

License

Professional use permitted with attribution.

⸻

Author: Richard Cheetham
Brand: Carolwood Estates · RACLuxe Division
Contact: info@racluxe.com

**a.** Want me to add small sample images/gifs to the v1.3 section with `--visualize` overlays?  
**b.** Add a “Requirements” badge row (FFmpeg / tifffile) under the main badges?