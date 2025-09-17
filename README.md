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

## License
Professional use permitted with attribution.
