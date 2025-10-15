# 800 Picacho Lane — Professional LUT Collection

## Overview

A cutting-edge collection of **16 professional color grading LUTs** featuring innovative **Material Response** technology.

## Quickstart

Install the package and run your first TIFF enhancement:

```bash
pip install picacho-lane-luts
python luxury_tiff_batch_processor.py input_folder output_folder --preset signature

## Table of Contents
- [Overview](#overview)
- [Collection Contents](#collection-contents)
- [Innovation](#innovation)
- [Usage](#usage)
    - [Material Response Finishing for Neural Renders](#material-response-finishing-for-neural-renders)
- [Developer Setup](#developer-setup)
    - [Install Dependencies](#install-dependencies)
    - [Test Shortcuts](#test-shortcuts)
- [Luxury TIFF Batch Processor](#luxury-tiff-batch-processor)
    - [TIFF Batch Processor Key Features](#tiff-batch-processor-key-features)
    - [TIFF Batch Processor Requirements](#tiff-batch-processor-requirements)
    - [TIFF Batch Processor Example](#tiff-batch-processor-example)
    - [Staying Synchronized With `main`](#staying-synchronized-with-main)
- [Luxury Video Master Grader](#luxury-video-master-grader)
    - [Luxury Video Master Grader Key Features](#luxury-video-master-grader-key-features)
    - [Luxury Video Master Grader Requirements](#luxury-video-master-grader-requirements)
    - [Luxury Video Master Grader Examples](#luxury-video-master-grader-examples)
    - [Replacing the Test Stub](#replacing-the-test-stub)
- [HDR Production Pipeline](#hdr-production-pipeline)
    - [HDR Production Pipeline Key Features](#hdr-production-pipeline-key-features)
    - [HDR Production Pipeline Highlights](#hdr-production-pipeline-highlights)
    - [HDR Production Pipeline Requirements](#hdr-production-pipeline-requirements)
    - [HDR Production Pipeline Example](#hdr-production-pipeline-example)
- [Decision Decay Dashboard](#decision-decay-dashboard)
- [License](#license)

## Collection Contents

- **Film Emulation**: Kodak 2393, FilmConvert Nitrate
- **Location Aesthetic**: Montecito Golden Hour, Spanish Colonial Warm
- **Material Response**: Revolutionary physics-based surface enhancement

## Innovation
**Material Response LUTs** analyze and enhance how different surfaces interact with light—shifting from purely global color transforms to surface-aware rendering that respects highlights, midtones, and micro-contrast differently across materials. The approach keeps natural roll-off in highlights, protects nuanced midtone skin detail, and intelligently refines specular textures so metal, stone, glass, and fabric all retain their distinctive depth.

### Design Considerations Beyond the Current Pipeline
- **Uncaptured sensory dimensions** – Acoustic character, scent associations, and haptic memory remain outside of the renderer's scope. Documenting these omissions helps stakeholders plan auxiliary storytelling layers (audio beds, scent marketing, tactile staging) that finish the luxury experience.
- **Procedural vs. authored control** – `lux_render_pipeline.py` mixes stochastic calls such as `algorithmic_stone`'s `weighted_random` selections with highly specific authored values (for example, hand-tuned fireplace glow radii). Teams should intentionally decide which elements benefit from parametric exploration and which require fixed creative direction so that variation never dilutes signature moments.
- **Modeling negative space** – While the pipeline meticulously models materials and lighting, it has no explicit "spatial restraint" metric (i.e., a guideline or measure for intentionally preserving empty space, minimalism, or the absence of visual clutter in a scene). When scenes demand quiet luxury, consider companion guidelines (shot lists, framing briefs, staging constraints) that preserve emptiness and silence instead of defaulting to additive richness.
- **Codification as creative baseline** – The system excels at making tacit craft repeatable. Treat that repeatability as a scaffold; the most memorable executions will still come from consciously breaking expectations once the technical baseline is in place.

## Usage

1. Import into DaVinci Resolve, Premiere Pro, or other color-grading software.
2. Apply at **60–80% opacity** initially.
3. Stack multiple LUTs for complex material interactions.

### Material Response Finishing for Neural Renders

`lux_render_pipeline.py` now exposes a **Material Response** finishing layer that reinforces
wood grain, textile separation, fireplace warmth, and atmospheric haze directly from the
command line. Enable it with `--material-response` to activate detail boosts, contact shadowing,
and volumetric tinting that better fuse interior renders with their exterior vistas.

Example:

```bash
python lux_render_pipeline.py \
  --input bedroom_render.jpg \
  --out ./enhanced_bedroom \
  --prompt "minimalist bedroom interior, wide plank oak flooring with visible grain and subtle sheen, white linen bedding with fabric texture, black leather tufted bench, linear gas fireplace casting warm glow, floor to ceiling windows, misty mountain lake view, morning light, photorealistic architectural visualization, material response technology, subtle ambient occlusion, volumetric lighting" \
  --neg "flat textures, uniform surfaces, no shadows, plastic looking, disconnected lighting, harsh contrast" \
  --width 1536 --height 864 \
  --steps 35 --strength 0.42 --gs 8.2 \
  --material-response --texture-boost 0.28 --ambient-occlusion 0.14 --highlight-warmth 0.1 --haze-strength 0.08 \
  --floor-plank-contrast 0.16 --floor-specular 0.22 --textile-contrast 0.24 \
  --floor-contact-shadow 0.06 \
  --leather-sheen 0.18 --fireplace-glow 0.26 --fireplace-glow-radius 60 \
  --window-reflection 0.18 --bedding-relief 0.22 --wall-texture 0.14 \
  --painting-integration 0.16 --window-light-wrap 0.2 --exterior-atmosphere 0.18 \
  --no-depth
```

Pair the result with the TIFF batch processor below for final client delivery.

Material Response finishing now maps common interior materials to targeted
micro-responses:

- **Floor plank contrast / specular** – reinforces oak grain variation, plank seams, and directional highlights flowing from the windows.
- **Floor contact shadow** – feathers a grounded penumbra at the floor transition even when global ambient occlusion is gentle.
- **Textile contrast** – separates linens, duvets, and throws with nuanced wrinkle shading so bedding reads tactile rather than plastic.
- **Leather sheen** – restores supple specular roll-off on benches and accent seating.
- **Fireplace glow** – diffuses a warm gradient from the flame box onto adjacent floors and walls for believable integration.
- **Window reflection** – projects soft mullion reflections and light spill onto the flooring while respecting ambient occlusion.
- **Bedding relief** – adds micro-wrinkle contrast and fold occlusion so duvets, throws, and pillows feel dimensional.
- **Wall texture** – injects subtle plaster/builder noise into bright painted surfaces so walls avoid a CG-flat read.
- **Painting integration** – pulls wall art into the room's lighting with rim glow and soft shadow falloff.
- **Window light wrap** – feathers panoramic daylight across floor, bed, and bench for cohesive lighting.
- **Exterior atmosphere** – harmonizes the exterior vista with interior haze for believable depth continuity.

The finishing pass now accepts authored texture plates and sky environments so renders inherit familiar materials from the
library. Supply the new CLI flags alongside `--material-response` to opt-in:

```bash
  --floor-texture ./textures/wide_oak.jpg --floor-texture-strength 0.75 --floor-texture-scale 0.8 \
  --wall-texture-path ./textures/limestone.jpg --wall-texture-strength 0.45 \
  --pool-texture-path ./textures/pool_mosaic.jpg --pool-texture-strength 0.6 \
  --sky-environment-path ./textures/coastal_dusk_sky.jpg --sky-environment-strength 0.8
```

Each texture flag includes a blend strength (0–1) and optional scale multiplier so you can match the plate to the render's
perspective without leaving the command line.

## Developer Setup

### Install Dependencies

- Create an isolated environment (optional but recommended):
  - `python -m venv .venv`
- Install the project runtime requirements:
  - `python -m pip install .` (installs core dependencies from pyproject.toml)
  - Alternatively: `python -m pip install -r requirements.txt` (to mirror CI)
- Add optional extras as needed:
  - `pip install -e ".[tiff]"` – 16-bit TIFF processing with tifffile
  - `pip install -e ".[dev]"` – pytest, pytest-xdist, flake8
  - `pip install -e ".[ml]"` – PyTorch, Diffusers, ControlNet for lux_render_pipeline
  - `pip install -e ".[all]"` – all optional dependencies

### Console Scripts

After installation with `pip install .`, the following command-line tools are available:

- `luxury-tiff-batch-processor` – batch process TIFF images with presets
- `luxury-video-master-grader` – apply LUTs and finishing to video files
- `lux-render-pipeline` – AI-powered render refinement (requires `[ml]` extras)
- `decision-decay-dashboard` – view temporal contracts and philosophy violations

You can also run the scripts directly with Python (e.g., `python -m luxury_tiff_batch_processor.cli`).

### Test Shortcuts

Use the bundled `Makefile` to run repeatable subsets during development:

- `make test-fast` – runs the material response suite plus the light-weight image processing tests.
- `make test-novideo` – executes every test except the FFmpeg-heavy video grader coverage.
- `make test-full` – runs the full suite, automatically parallelising with `pytest-xdist` when available.

### Code Quality

The repository uses both **flake8** and **pylint** for code quality checks:

- **flake8**: Enforces critical syntax and import checks (configured in CI)
  ```bash
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  ```
- **pylint**: Comprehensive code analysis with project-specific configuration
  ```bash
  pylint $(git ls-files '*.py')
  ```
  Configuration in `.pylintrc` aligns with project coding standards (127 char line length, selective documentation, etc.). CI passes with a score threshold of 9.0/10.

## Luxury TIFF Batch Processor
The repository now includes `luxury_tiff_batch_processor.py`, a high-end batch workflow for polishing large-format TIFF photography prior to digital launch. The script preserves metadata, honors 16-bit source files when [`tifffile`](https://pypi.org/project/tifffile/) is available, and layers tonal, chroma, clarity, and diffusion refinements tuned for ultra-luxury real-estate storytelling.

### TIFF Batch Processor Key Features
- Automatically reads and writes IPTC/XMP metadata so campaign details remain intact across exports.
- Maintains 16-bit precision whenever the optional `tifffile` dependency is installed and falls back to Pillow for 8-bit output.
- Offers presets that mirror the LUT families (Signature, Golden Hour, Heritage, etc.) for rapid client alignment.
- Supports per-run overrides for exposure, midtone contrast, vibrance, clarity, glow, and more to accommodate creative direction.
- Provides non-destructive previews with `--dry-run` and mirrors directory trees with `--recursive` for large productions.
- Scales across CPU cores with `--workers` for accelerated delivery while preserving dry-run and overwrite safeguards.

### TIFF Batch Processor Requirements
- Python 3.11+
- Install dependencies with `pip install .` (or `pip install -r requirements.txt` to mirror CI).
- Add `tifffile` for lossless 16-bit output.

> **Note:** Earlier revisions triggered `F821` undefined-name lint errors. Pull the latest main branch (or reinstall from the freshest ZIP) to ensure you have the corrected helper that resolves the NumPy dtype handling.

### Example (TIFF Batch Processor)
```bash
python luxury_tiff_batch_processor.py /path/to/raw_tiffs /path/to/output \
  --preset signature --resize-long-edge 7000 --overwrite
```

#### Processing profiles

The TIFF pipeline now offers lightweight processing profiles so you can tune
fidelity against turnaround directly from the CLI:

- `quality` (default) keeps every adjustment active and preserves the source
  bit depth and compression choices for maximum latitude.
- `balanced` retains most tonal work but applies a softened glow/denoise pass
  and promotes output to 16-bit precision when available so files grade well
  without the full render cost of the quality profile.
- `performance` disables the most expensive spatial filters, targets an
  8-bit export, and switches to JPEG-compressed TIFF output for extremely fast
  client previews.

Select a profile with `--profile quality|balanced|performance` or set it in a
configuration file. All presets and manual overrides continue to work, while
the profile determines how aggressively the high-cost filters run and the
bit-depth/compression of the exported files.

#### Configuration files

Every command-line flag (other than `--config` itself) can be provided through a
JSON or YAML configuration file. JSON is supported out of the box, while YAML
files require [`pyyaml`](https://pypi.org/project/PyYAML/) to be installed.

Create a configuration file with keys that mirror the argument names (hyphens
and underscores are treated interchangeably):

```json
{
  "input": "./input_images",
  "output": "./processed_images",
  "preset": "signature",
  "recursive": true,
  "resize_long_edge": 7000,
  "workers": 4
}
```

Or, when `pyyaml` is available, YAML:

```yaml
input: ./input_images
output: ./processed_images
preset: heritage
recursive: true
suffix: _client
log_level: DEBUG
```

Load the configuration and optionally override specific settings on the command
line:

```bash
python luxury_tiff_batch_processor.py --config settings.json --workers 8 --overwrite
```

Values specified on the command line always take precedence over configuration
defaults, and unknown configuration keys raise a helpful error so typos are
caught quickly.

For coastal twilight exteriors that need richer terracotta, nuanced lawn depth, and
Pacific color separation, start from the new **Golden Hour Courtyard** preset and layer in
the creative overrides recommended by the Material Response team.

> **Note:** Although the preset is named "Golden Hour Courtyard," it has been optimized to work well for coastal twilight exteriors when combined with the recommended creative overrides below. This approach leverages the preset's material response characteristics to achieve the desired look in twilight conditions.
```bash
python luxury_tiff_batch_processor.py input_folder output_folder \
  --preset golden_hour_courtyard \
  --exposure 0.08 --shadow-lift 0.24 --highlight-recovery 0.18 \
  --vibrance 0.28 --clarity 0.20 --luxury-glow 0.12 --white-balance-temp 5600 \
  --midtone-contrast 0.10
```

When an explicit output folder is not provided the processor now creates a sibling
directory named after the input with a `_lux` suffix. For example, running:

```bash
python luxury_tiff_batch_processor.py /Users/rc/Downloads/Montecito-Shores-2/SV-Stills
```

will mirror the processed files in `/Users/rc/Downloads/Montecito-Shores-2/SV-Stills_lux`.

Override any preset control (exposure, midtone contrast, vibrance, clarity, glow, etc.)
by providing the corresponding command-line flag. Use `--dry-run` to audit the processing
plan without writing files, and `--recursive` to mirror nested shoot-day folders.

### Staying Synchronized With `main`
To keep feature branches review-ready, regularly reconcile them with the newest
`main` history:

1. `git fetch origin`
2. `git checkout main`
3. `git pull`
4. `git checkout <your-branch>`
5. `git merge origin/main` *(or `git rebase origin/main` if you prefer a linear history)*

Resolve any conflicts, rerun your tests, then push the refreshed branch. GitHub
will report `behind 0` once these steps are complete.

## Luxury Video Master Grader

`luxury_video_master_grader.py` brings the same curated aesthetic to short-form motion content. It wraps FFmpeg with preset-driven LUT application, tasteful denoising, clarity and film-grain treatments, then exports a mezzanine-ready Apple ProRes master by default. The pipeline now auto-detects HDR transfers and tone maps them into a refined BT.709 space, optionally adds ultra-fine debanding and cinematic halation bloom, and keeps gradient-rich interiors spotless with updated presets.

### Luxury Video Master Grader Key Features
- Intelligent source analysis that detects resolution, frame rate, HDR transfer characteristics, and audio layout before rendering.
- Preset-driven workflows that blend LUT application with spatial/temporal filtering tailored to luxury real-estate cinematography.
- Advanced finishing controls for debanding, halation, clarity, grain, and tone mapping to deliver cinematic masters out of the box.
- Flexible output targets including mezzanine-grade ProRes profiles and user-defined frame rates via `--target-fps`.
- Dry-run inspection that prints the underlying FFmpeg command for validation before committing to long renders.

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

Use `--custom-lut` to feed bespoke `.cube` files, tweak parameters such as `--contrast` or `--grain`, layer in `--deband` smoothing or halation controls, and enable `--dry-run` to inspect the underlying FFmpeg command without rendering. The script automatically probes the source to surface resolution, frame-rate metadata and audio configuration before processing, then monitors for drift or variable frame-rate clips. When necessary it conforms delivery to the nearest cinema broadcast standard (or a user-specified `--target-fps`) to guarantee smooth, continuous playback. HDR clips are further analyzed so the tool can apply tasteful tone mapping automatically, while explicit `--tone-map` overrides give you authoritative control whenever a specific operator is required.

### Replacing the Test Stub

The repository bundles a minimal `luxury_video_master_grader.py` so the automated tests can validate CLI glue even when the production module is unavailable. To exercise your real grader instead:

1. Install the production package that provides the `luxury_video_master_grader` import in the active environment.
2. Rename or remove the repository stub (it lives at the repo root) so Python resolves the installed package first.
3. Confirm the import path: `python -c "import luxury_video_master_grader, inspect; print(luxury_video_master_grader.__file__)"`.
4. Run `make test-full` (or `pytest -q`) to ensure the video tests execute against the real implementation.

## HDR Production Pipeline

`hdr_production_pipeline.sh` orchestrates a full HDR finishing pass, combining ACES tone mapping, adaptive debanding, and filmic halation for gallery-ready masters. The workflow harmonizes the bespoke Codex automation steps with the broader pipeline overview introduced on main so teams can reference a single, unified set of HDR finishing instructions.

### HDR Production Pipeline Key Features
- ACES Output Device Transform (ODT) selection, Dolby Vision metadata pass-through, and HDR10 mastering options for broadcast compliance.
- Adaptive debanding tuned to Codex reference recipes to protect smooth gradients in modern architectural interiors.
- Filmic halation and bloom controls that layer naturally over the LUT aesthetic without introducing color drift.
- Optional tone-mapping operator selection (`--tone-map`) for precise control over HDR-to-SDR conversions.
- Designed to slot after `luxury_video_master_grader.py` so teams can reuse LUT-driven looks while finishing in HDR.

### HDR Production Pipeline Highlights
- Consolidates automation steps from the Codex finishing toolkit into a single, reproducible command sequence.
- Documents the hand-off from the LUT-driven grade to HDR-specific finishing so teams can collaborate without guesswork.
- Provides defaults that mirror the examples in this README, making it simple to reproduce the reference masters.

### HDR Production Pipeline Requirements
- macOS or Linux shell environment
- FFmpeg with zimg, `ffprobe`, and `python3`

### HDR Production Pipeline Example
```bash
./hdr_production_pipeline.sh source_hdr.mov out_hdr_master.mov \
  --aces-odt rec2020-pq --deband medium --halation strong --hdr10-metadata auto
```

Layer the script after `luxury_video_master_grader.py` to apply bespoke LUTs before the HDR-specific finishing tools run. The pipeline preserves Dolby Vision and static HDR10 metadata where available, while the deband and halation stages default to the Codex branch recipes highlighted in the documentation examples.

## Board Material Aerial Enhancer

`board_material_aerial_enhancer.py` applies MBAR (Montecito Board of Architectural Review) approved material palettes to aerial photographs using k-means color clustering and texture blending. The system identifies architectural surfaces (plaster, stone, wood cladding, metal panels) and applies high-resolution material textures for board-ready deliverables.

### Key Features
- **Automatic Material Detection**: HSV-based heuristics identify 8 MBAR-approved materials
- **Palette Configuration**: Save and load cluster-to-material assignments as JSON files
- **Deterministic Workflows**: Reuse palette files for consistent results across multiple aerials
- **4K Output**: Scales enhanced images to 4096px width for presentation materials
- **Texture Blending**: Soft Gaussian masks ensure natural material transitions

### Palette Assignment Workflow

The enhancer now supports **palette files** - JSON configurations that map cluster IDs to specific materials. This enables manual override of automatic detection and deterministic assignments:

```bash
# Step 1: Generate automatic assignments and save palette
python board_material_aerial_enhancer.py \
  aerial.jpg enhanced.jpg \
  --k 8 \
  --save-palette auto_palette.json

# Step 2: Review and edit auto_palette.json to adjust materials
# Example: Change cluster 3 from "stone" to "plaster"

# Step 3: Apply refined palette to all project aerials
python board_material_aerial_enhancer.py \
  aerial_view_2.jpg enhanced_view_2.jpg \
  --palette auto_palette.json
```

Palette JSON format:
```json
{
  "version": "1.0",
  "assignments": {
    "0": "plaster",
    "1": "stone",
    "2": "roof",
    "3": "equitone"
  }
}
```

**Available Materials**: `plaster`, `stone`, `cladding`, `screens`, `equitone`, `roof`, `bronze`, `shade`

For complete documentation, see [Palette Assignment Guide](08_Documentation/Palette_Assignment_Guide.md).

### Basic Usage

```bash
python board_material_aerial_enhancer.py input.jpg output.jpg
```

### Advanced Options

```bash
python board_material_aerial_enhancer.py \
  input.jpg output.jpg \
  --analysis-max 1280 \
  --k 8 \
  --seed 22 \
  --target-width 4096 \
  --palette custom_palette.json \
  --save-palette computed_palette.json
```

**Options**:
- `--analysis-max`: Max dimension for clustering (default: 1280)
- `--k`: Number of k-means clusters (default: 8)
- `--seed`: Random seed for reproducibility (default: 22)
- `--target-width`: Output width in pixels (default: 4096)
- `--palette`: Load cluster assignments from JSON file
- `--save-palette`: Save computed assignments to JSON file

The enhancer includes a visualization tool for reviewing material assignments:

```bash
python visualize_material_assignments.py --save-palette viz_palette.json
```

## Decision Decay Dashboard

`decision_decay_dashboard.py` surfaces temporal contracts, codebase philosophy violations, and brand color token drift in a single terminal dashboard. It cross-references `tests/` for `@valid_until` decorators, audits Python sources with `CodebasePhilosophyAuditor`, and highlights unused color tokens from the Lantern logo deliverables so teams know what requires attention next.

### Running the dashboard
```bash
python decision_decay_dashboard.py
```

Use `--root` to audit a different project tree, `--tests` or `--tokens` to point at alternate assets, and `--json <path>` to export the findings for downstream automation. Near-term `valid_until` expirations are flagged in yellow when `rich` is installed (or with a `!` prefix in plain text). Philosophy violations are aggregated by principle with example file locations, while the color section lists which brand hex values are still unused in CSS/JS deliverables.

## License

Professional use permitted with attribution.
