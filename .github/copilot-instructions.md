```markdown
# Copilot Instructions for 800 Picacho Lane LUTs

## Project Overview
Professional LUT (Look-Up Table) collection and image/video processing toolchain for luxury real estate cinematography and photography.

- **LUT Collection**: Material Response film/location LUTs (Kodak emulations, location aesthetics, material-aware rendering)
- **Python Tools**: TIFF batch processing, video master grading, AI render refinement
- **Shell Tools**: HDR finishing pipeline
- **ML Pipeline**: Stable Diffusion + ControlNet orchestration

---

## Repository Structure

```

.
├── 01_Film_Emulation/                  # Film LUTs (Kodak, FilmConvert-style)
├── 02_Location_Aesthetic/              # Location-specific color profiles
├── 03_Material_Response/               # Physics/material-aware LUTs
├── 08_Documentation/                   # Version history & tech guides
├── 09_Client_Deliverables/             # Brand assets & deliverables
├── tests/                               # pytest test suite (100+)
├── luxury_tiff_batch_processor.py       # Legacy CLI (see package CLIs too)
├── luxury_video_master_grader.py        # Video grading with FFmpeg
├── lux_render_pipeline.py               # AI render refinement
├── hdr_production_pipeline.sh           # HDR finishing workflow
├── codebase_philosophy_auditor.py       # Code quality auditing
├── decision_decay_dashboard.py          # Temporal contract monitoring
└── src/                                 # Modernized modules (e.g., evolutionary.py)

````

> **Note**: Some CLIs live under package modules and expose Typer commands; see `--help` on each script/entrypoint.

---

## Getting Started

### Requirements
- Python **3.10+** (CI matrix: 3.10/3.11/3.12)
- FFmpeg **6+** for video tools
- Git

### Install (quick)
```bash
git clone https://github.com/RC219805/800-Picacho-Lane-LUTs.git
cd 800-Picacho-Lane-LUTs

# Project (editable) + lean dev extras typically used in CI
pip install -e ".[dev-lean]"
````

### Install (CI-parity)

```bash
# Pins for pytest-cov, flake8 7, pylint 3, mypy 1.x, etc.
pip install -r requirements-ci.txt
pip install -e ".[dev-lean]"
```

### Verify

```bash
# Shim guard + sanity imports (matches CI steps)
python tools/gen_legacy_shims.py --fail-on-create
python - <<'PY'
import importlib; importlib.import_module('src.evolutionary'); importlib.import_module('evolutionary_checkpoint'); print("sanity OK")
PY
```

---

## Common Tasks

### Run tests

```bash
pytest -v
# With coverage (CI-style)
pytest -v --cov=. --cov-report=xml --cov-report=term-missing
```

### Lint / Type check

```bash
# Fast flake8 errors-only gate (matches CI)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Pylint (threshold aligned with CI workflows)
pylint -j 0 --fail-under=8.0 .

# MyPy (exclude heavy dirs as needed)
mypy .
```

### Generate/commit shims (only when needed)

If CI fails with the shim guard (`--fail-on-create`), pre-generate and commit:

```bash
# Write missing shims
python tools/gen_legacy_shims.py --write
git add *.py
git commit -m "ci(shims): add/update legacy shims"
```

> Or run the manual **shim-sync** workflow from Actions.

---

## Tech Stack

* **Python**: 3.10–3.12
* **Core libs**: NumPy, Pillow, tifffile, imagecodecs, Typer, tqdm
* **Video**: FFmpeg 6+ (ProRes masters, HDR handling)
* **ML** (optional extras): PyTorch, Diffusers, ControlNet, RealESRGAN
* **Quality**: pytest (+pytest-cov), flake8 7, pylint 3, mypy 1.8+

---

## Coding Standards

* PEP 8; max line length **127** (per CI)
* Type hints where practical; prefer `pathlib.Path` and f-strings
* Dataclasses for configs; focused, single-purpose functions
* Document *why* when behavior is non-obvious; keep function docs concise
* File names: `snake_case.py`; tests as `tests/test_<module>.py`

---

## LUT & Processing Guidelines

### LUTs

* `.cube` files organized under `01_*`, `02_*`, `03_*`
* Typical strength: **60–80%**
* Stacking allowed for complex material interactions

### Video (FFmpeg)

* Default output: **ProRes 422 HQ** masters
* HDR detection + tone mapping; preserve metadata
* Frame rate conformance; preset-driven workflows (e.g., `signature_estate`)

### Images (TIFF)

* Preserve **16-bit** precision where possible
* Maintain IPTC/XMP metadata
* Batch with tree mirroring; preset-based adjustments (exposure, contrast, sat, clarity, glow)

---

## Adding / Modifying

### Add a new preset (example)

```python
# In luxury_video_master_grader.py (or relevant CLI)
PRESETS["my_new_preset"] = PresetConfig(
    name="My New Preset",
    lut="01_Film_Emulation/Kodak_2393.cube",
    notes="Custom look for specific project",
    exposure=0.0,
    contrast=1.08,
    saturation=1.05,
    clarity=0.15,
    grain=0.012,
)
```

### Update FFmpeg filters

* Use builder patterns (`build_filter_graph()`); validate via `--dry-run`
* Preserve color metadata (`color_primaries`, `color_trc`, `colorspace`)
* Test SDR and HDR sources

### Add CLI options

* Use **Typer**; group related options; provide sensible defaults
* Always support `--dry-run` for inspection

---

## Testing Guidelines

* Place tests under `tests/`; use fixtures for shared setup
* Mock external tools (FFmpeg, filesystem) where sensible
* Edge cases: missing files, invalid parameters, HDR content
* Keep tests deterministic (seeded randomness)
* Run `pytest -v` locally before PRs

---

## CI/CD (what to expect)

* Workflows:

  * **build.yml**: matrix tests + coverage + mypy + shim guard + sanity imports + Codecov
  * **ci.yml**: lean, fast green/red with shim guard
  * **static-analysis.yml**: Black/isort/Flake8/Pylint, mypy cached, Bandit, Radon
  * **pylint.yml**: standalone lint + mypy
  * **codeql.yml / codeql-advanced.yml**: security scans
  * **shim-sync.yml**: manual generator to *write/commit* shims
  * **summary.yml**: AI issue summarizer (requires `OPENAI_API_KEY`)
* Pins live in `requirements-ci.txt`; CI uses `pip install -r requirements-ci.txt` + `-e ".[dev-lean]"`.
* Failing imports? CI runs a **sanity import** step; fix by adding missing modules or shims.

---

## When Making Changes

1. Validate the pipeline order (color/HDR transforms are order-dependent)
2. Test with real media when available
3. Use `--dry-run` for long FFmpeg flows
4. Avoid breaking existing CLI flags or outputs
5. Update docs/examples
6. Preserve metadata (IPTC/XMP/GPS) wherever possible

---

## Troubleshooting

**Import errors (`evolutionary_checkpoint`)**

* Ensure `src/evolutionary.py` and `src/__init__.py` are present
* Run `python tools/gen_legacy_shims.py --fail-on-create` (CI gate)
* If needed: `python tools/gen_legacy_shims.py --write` then commit

**`pytest: unrecognized arguments --cov`**

* Install `pytest-cov>=4,<5` (`requirements-ci.txt`)

**Pylint `--logging-format-style` error**

* Use `--logging-format-style=old` (Pylint 3)

**FFmpeg issues**

* Inspect via `--dry-run`, verify LUT paths, check HDR support (`ffmpeg -filters | grep zscale`)

**Differences CI vs local**

* Align Python version and `requirements-ci.txt`
* Check env/timezone; Linux paths are case-sensitive

Quick debug:

```bash
python --version
pip list
env | sort | sed -n '1,40p'
```

---

## PR Checklist (fast)

* [ ] `pytest -v` (and with coverage if relevant)
* [ ] `flake8` errors-only gate
* [ ] `pylint --fail-under=8.0`
* [ ] `mypy .`
* [ ] `python tools/gen_legacy_shims.py --fail-on-create`
* [ ] Docs/examples updated if user-facing behavior changed

---
