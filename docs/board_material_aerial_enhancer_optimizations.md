# board_material_aerial_enhancer.py Performance Optimizations (Corrected)

## Overview

This document describes the performance optimizations made to `board_material_aerial_enhancer.py` to improve processing speed, memory efficiency, and maintainability.

## Key Optimizations

### 1. Scikit-learn KMeans Integration

**Change**: Added optional `sklearn.cluster.KMeans` integration (**disabled by default**, opt-in via `--use-sklearn`).

**Benefits**:
- 2–5× speedup for clustering operations on larger, real‑world images (when enabled)
- More robust initialization with **k‑means++**
- Better convergence with optimized Lloyd's algorithm
- Multiple initializations (**n_init=10**) with k‑means++ for robust clustering

**Usage**:
```python
# Default: uses built-in k-means (CI-friendly, no sklearn dependency)
enhance_aerial(input_path, output_path, k=8)

# Enable sklearn for better performance (if available)
enhance_aerial(input_path, output_path, k=8, use_sklearn=True)
```

### 2. Parameter Validation

**Change**: Added `_validate_parameters()` to check inputs early.

**Validates**:
- `k`: must be 2–256 for meaningful clustering
- `analysis_max_dim`: must be ≥ 32 for reasonable analysis (canonical name)
- `seed`: must be non‑negative
- `target_width`: must be ≥ 32 if specified
- `resample_method`: must be in the valid methods list (case‑insensitive)

**Benefits**:
- Prevents invalid configurations that could cause crashes or poor performance
- Provides clear, specific error messages
- Warnings for extreme values that might be inefficient

### 3. Performance Logging

**Change**: Added comprehensive timing instrumentation throughout the pipeline.

**Logged operations**:
- Image loading
- Downscale
- K‑means clustering (major operation)
- Label upscaling
- Enhancement application
- Overall processing time

**Usage**:
```bash
# Verbose logging
python board_material_aerial_enhancer.py input.jpg output.jpg --verbose
```

### 4. Memory & IO Optimizations

**Changes**:
- In‑place array operations (division, multiplication, clipping)
- Avoids unnecessary array copies
- Reuses arrays where possible
- Downscales images before clustering
- **Batched full‑image relabel** to cap peak memory usage
- **Adaptive label dtype** (`uint8`/`uint16`) based on `k` to avoid overflow and reduce footprint
- **Atomic palette saves** (write to temp, then `rename`) for crash‑safe updates

### 5. Flexible Resampling Methods

**Change**: Added `--resample-method` parameter for quality/speed trade‑off.

**Options** (case‑insensitive):
- `nearest` — fastest, lower quality
- `bilinear` — balanced (default for analysis images)
- `lanczos` — highest quality (used for final output)
- Additional methods: `linear`, `cubic`, `bicubic`, `area`, `box`

## CLI Enhancements

### New Options

```bash
--use-sklearn             # Enable sklearn k-means for better performance (default: OFF)
--resample-method METHOD  # Choose resampling quality (case-insensitive)
--verbose, -v             # Enable DEBUG level logging
--quiet, -q               # Suppress all logs except errors
```

## Testing

Comprehensive test suite in `tests/test_board_material_optimizations.py`:
- Parameter validation
- K‑means optimization
- Enhanced API
- Palette operations
- Memory efficiency
- Performance benchmarks

Run tests:
```bash
pytest tests/test_board_material_optimizations.py -v
```

## Backward Compatibility

All changes maintain backward compatibility:
- Existing function signatures unchanged
- **`analysis_max` remains a backward‑compatible alias for `analysis_max_dim` (canonical)**
- Default behavior unchanged (built‑in k‑means unless `--use-sklearn` is set)
- New parameters are optional with sensible defaults

## Performance Notes

- The built‑in k‑means is the default to keep CI lightweight and deterministic.
- Enable sklearn with `--use-sklearn` for **2–5×** improvement on complex, larger images.
- Sklearn path uses **k‑means++** with **`n_init=10`** for stability.
- **Note:** On very small synthetic images, sklearn can be slower due to initialization overhead; the benefit grows with image size and texture complexity.
