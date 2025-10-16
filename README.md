[![CI](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions/workflows/python-app.yml/badge.svg)](https://github.com/RC219805/800-Picacho-Lane-LUTs/actions)
[![License](https://img.shields.io/badge/license-Attribution-blue.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/build-passing-success.svg)](https://github.com/RC219805/800-Picacho-Lane-LUTs)

# 800 Picacho Lane — Professional LUT Collection

## Overview
A cutting-edge collection of **16 professional color grading LUTs** featuring innovative **Material Response** technology.

## Quickstart
Install the package and run your first TIFF enhancement:
```bash
pip install picacho-lane-luts
python luxury_tiff_batch_processor.py input_folder output_folder --preset signature
```

For advanced render finishing, see [Material Response Finishing for Neural Renders](#material-response-finishing-for-neural-renders).

---

## Table of Contents

* [Overview](#overview)
* [Quickstart](#quickstart)
* [Collection Contents](#collection-contents)
* [Innovation](#innovation)
* [Usage](#usage)

  * [Material Response Finishing for Neural Renders](#material-response-finishing-for-neural-renders)
* [Developer Setup](#developer-setup)

  * [Install Dependencies](#install-dependencies)
  * [Test Shortcuts](#test-shortcuts)
* [Luxury TIFF Batch Processor](#luxury-tiff-batch-processor)
* [Luxury Video Master Grader](#luxury-video-master-grader)
* [HDR Production Pipeline](#hdr-production-pipeline)
* [Board Material Aerial Enhancer](#board-material-aerial-enhancer)
* [Decision Decay Dashboard](#decision-decay-dashboard)
* [License](#license)

---

## Collection Contents

* **Film Emulation** — Kodak 2393, FilmConvert Nitrate
* **Location Aesthetic** — Montecito Golden Hour, Spanish Colonial Warm
* **Material Response** — Revolutionary physics-based surface enhancement

---

## Innovation

**Material Response LUTs** analyze and enhance how different surfaces interact with light—shifting from global transforms to **surface-aware** rendering that respects highlights, midtones, and micro-contrast differently across materials.

---

## Usage

1. Import into DaVinci Resolve, Premiere Pro, or any LUT-capable grading suite.
2. Apply at **60–80% opacity** for best results.
3. Stack multiple LUTs for complex surface interactions.

### Material Response Finishing for Neural Renders

[`lux_render_pipeline.py`](./lux_render_pipeline.py) adds a **Material Response** finishing layer that enhances wood grain, textiles, fireplace glow, and atmospheric haze.
Enable with `--material-response` for physically consistent detail boosts and volumetric tinting.

> **Requires:** `[ml]` extras (`pip install -e ".[ml]"`) and a GPU-enabled environment.

Example:

```bash
python lux_render_pipeline.py   --input bedroom_render.jpg   --out ./enhanced_bedroom   --prompt "minimalist bedroom interior..."   --material-response --texture-boost 0.28
```

---

## Developer Setup

### Install Dependencies

```bash
python -m pip install .
# or mirror CI
python -m pip install -r requirements.txt
```

Optional extras:

```bash
pip install -e ".[tiff]"   # 16-bit TIFF processing
pip install -e ".[dev]"    # pytest, linting
pip install -e ".[ml]"     # ML extras for lux_render_pipeline
pip install -e ".[all]"    # everything
```

### Console Scripts

After installation, the following CLI tools are available:

* [`luxury_tiff_batch_processor.py`](./luxury_tiff_batch_processor.py) — batch process TIFFs
* [`luxury_video_master_grader.py`](./luxury_video_master_grader.py) — video grading
* [`lux_render_pipeline.py`](./lux_render_pipeline.py) — AI-powered render refinement
* [`decision_decay_dashboard.py`](./decision_decay_dashboard.py) — codebase philosophy audits

### Test Shortcuts

```bash
make test-fast
make test-full
```

`make ci` runs linting + fast tests (mirrors GitHub Actions).

---

## Luxury TIFF Batch Processor

[`luxury_tiff_batch_processor.py`](./luxury_tiff_batch_processor.py) handles **16-bit TIFFs** with color-accurate tonal adjustments and metadata preservation—perfect for high-end marketing imagery.

---

## Luxury Video Master Grader

[`luxury_video_master_grader.py`](./luxury_video_master_grader.py) applies the same cinematic tone and texture treatment to video content using FFmpeg.

---

## HDR Production Pipeline

[`hdr_production_pipeline.sh`](./hdr_production_pipeline.sh) orchestrates full HDR finishing, combining ACES tone mapping, adaptive debanding, and halation glow.

---

## Board Material Aerial Enhancer

[`board_material_aerial_enhancer.py`](./board_material_aerial_enhancer.py) applies MBAR-approved material palettes to aerials via optimized clustering and texture blending.

### **New Performance Features**

* Optional **scikit-learn KMeans** integration for 2–5× faster clustering (`--use-sklearn`)
* Strict **parameter validation** to prevent invalid configurations
* **Comprehensive logging** and timing instrumentation
* **Memory optimization** using in-place operations
* **Flexible resampling** for quality/speed trade-offs

### **Basic Usage**

```bash
# Quick enhancement with defaults
python board_material_aerial_enhancer.py input.jpg output.jpg

# Verbose logging and custom params
python board_material_aerial_enhancer.py input.jpg output.jpg   --k 12 --analysis-max 2048 --target-width 4096 --verbose
```

### **Performance Options**

```bash
# Fast preview (lower quality)
python board_material_aerial_enhancer.py input.jpg output.jpg   --resample-method NEAREST --k 4

# High-quality output
python board_material_aerial_enhancer.py input.jpg output.jpg   --resample-method LANCZOS --k 16
```

### **Advanced Flags**

```bash
# Enable the sklearn KMeans path (optional speed-up; OFF by default)
python board_material_aerial_enhancer.py input.jpg output.jpg --use-sklearn

# Reproducibility
python board_material_aerial_enhancer.py input.jpg output.jpg --seed 22

# Palette IO
python board_material_aerial_enhancer.py input.jpg output.jpg   --palette ./palette.json --save-palette ./out_palette.json

# Quiet logging (errors only)
python board_material_aerial_enhancer.py input.jpg output.jpg --quiet
```

**Notes:**

* `--use-sklearn` toggles the sklearn clustering path (**default is off**) and can significantly speed up larger, real‑world images.
* On very small synthetic images, the sklearn path can be slower due to initialization overhead; the benefit grows with image size and texture complexity.
* `--seed` ensures deterministic cluster assignments for reproducible runs.
* `--palette` / `--save-palette` provide material mapping persistence across projects.
* `--quiet` suppresses log output (pair with `--verbose` for full diagnostic mode).

For full documentation, see:

* [Optimization Guide](./docs/board_material_aerial_enhancer_optimizations.md)
* [Palette Assignment Guide](./08_Documentation/Palette_Assignment_Guide.md)

---

## Decision Decay Dashboard

[`decision_decay_dashboard.py`](./decision_decay_dashboard.py) visualizes architectural and stylistic drift across codebases—monitoring time decay in technical decisions and brand cohesion.

---

## License

Professional use permitted with attribution.

---

**Author:** Richard Cheetham  
**Brand:** Carolwood Estates · RACLuxe Division  
**Contact:** [info@racluxe.com](mailto:info@racluxe.com)
