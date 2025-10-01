# Copilot Instructions for 800 Picacho Lane LUTs

## Project Overview

This repository contains a professional LUT (Look-Up Table) collection and video/image processing toolchain for luxury real estate cinematography and photography. The project includes:

- **LUT Collection**: 16 professional color grading LUTs featuring Material Response technology for Kodak film emulation, location aesthetics, and material-aware rendering
- **Python Processing Tools**: Batch processors for TIFF images and video master grading
- **Shell Scripts**: HDR production pipeline for broadcast-ready masters
- **Machine Learning Pipeline**: AI-powered render refinement using Stable Diffusion and ControlNet

## Tech Stack

- **Languages**: Python 3.11+, Shell (Bash)
- **Key Dependencies**: 
  - FFmpeg 6+ (for video processing)
  - NumPy, Pillow, tifffile (for image processing)
  - PyTorch, Diffusers, ControlNet (for ML pipeline)
  - Typer (for CLI applications)
- **Testing**: pytest
- **Linting**: flake8

## Coding Standards

### Python
- Follow PEP 8 style guidelines
- Maximum line length: 127 characters (as configured in CI)
- Use type hints where appropriate
- Use dataclasses for configuration objects
- Prefer pathlib.Path over string paths
- Use f-strings for string formatting

### Code Organization
- Keep functions focused and single-purpose
- Use descriptive variable names (e.g., `preset`, `filter_graph`, `tone_map_config`)
- Document complex algorithms with docstrings
- Separate concerns: CLI parsing, business logic, and I/O operations

### File Naming
- Python scripts: lowercase with underscores (e.g., `luxury_video_master_grader.py`)
- Shell scripts: lowercase with underscores (e.g., `hdr_production_pipeline.sh`)
- Test files: `test_` prefix (e.g., `test_luxury_video_master_grader.py`)

## Key Concepts

### LUT Processing
- LUTs are stored in `.cube` format in categorized directories:
  - `01_Film_Emulation/` - Kodak and FilmConvert emulations
  - `02_Location_Aesthetic/` - Location-specific color profiles
  - `03_Material_Response/` - Physics-based surface enhancement
- LUT strength is typically applied at 60-80% opacity
- LUTs can be stacked for complex material interactions

### Video Processing
- Default output: ProRes 422 HQ masters
- HDR detection and automatic tone mapping
- Frame rate conformance to cinema/broadcast standards
- Preset-driven workflows (e.g., `signature_estate`, `golden_hour_courtyard`)

### Image Processing
- Preserve 16-bit TIFF precision when `tifffile` is available
- Maintain IPTC/XMP metadata across processing
- Support for batch processing with directory tree mirroring
- Preset-based adjustments for exposure, contrast, saturation, clarity, glow

## Testing Guidelines

- Write unit tests for new functions in the `tests/` directory
- Use pytest fixtures for common setup
- Mock external dependencies (FFmpeg, file I/O) when appropriate
- Test edge cases: missing files, invalid parameters, HDR content
- Ensure tests pass with `pytest` before committing

## Common Tasks

### Adding a New Preset
1. Define preset in the appropriate script (e.g., `PRESETS` dict in `luxury_video_master_grader.py`)
2. Include descriptive name, LUT path, and parameter values
3. Add documentation in the preset's `notes` field
4. Update README with example usage

### Modifying FFmpeg Filters
1. Use the `build_filter_graph()` pattern to construct filter chains
2. Always validate filter syntax with `--dry-run` option
3. Preserve color metadata (`color_primaries`, `color_trc`, `colorspace`)
4. Test with both SDR and HDR sources

### Adding CLI Options
1. Use Typer for Python CLIs (consistent with existing scripts)
2. Provide helpful descriptions and sensible defaults
3. Group related options with comments (e.g., `# Render config`, `# Models`)
4. Support `--dry-run` for inspection without execution

## Repository-Specific Notes

### HDR Handling
- Automatically detect HDR transfer functions (PQ, HLG)
- Apply tone mapping with configurable operators (Hable, Reinhard, Mobius)
- Preserve HDR10 and Dolby Vision metadata when available
- Use ACES ODT for broadcast compliance

### Performance Considerations
- Video processing can be GPU-intensive (CUDA/MPS support)
- Batch operations should report progress
- Use multiprocessing for independent image operations
- Provide early validation before long-running operations

### Brand/Client Deliverables
- Support for logo overlays (PNG with alpha)
- Maintain GPS coordinates in metadata when available
- Follow naming conventions: `{basename}_{preset}.{ext}`
- Create timestamped output directories for productions

## Documentation

- Keep README.md synchronized with tool capabilities
- Update `08_Documentation/Version_History/changelog.md` for significant changes
- Include usage examples with common parameter combinations
- Document all presets with their intended use cases

## CI/CD

- GitHub Actions runs tests on push and PR
- Flake8 linting enforces code quality
- Tests must pass before merging
- Python 3.10 is the CI target (ensure 3.11+ compatibility)

## When Making Changes

1. **Understand the pipeline**: Video/image processing pipelines are complex and order-dependent
2. **Test with real files**: Use sample LUTs and media files when available
3. **Validate FFmpeg commands**: Use `--dry-run` to inspect generated commands
4. **Consider backward compatibility**: Existing scripts may be in production use
5. **Update documentation**: Keep README and examples current
6. **Preserve metadata**: IPTC, XMP, and GPS data should survive processing
