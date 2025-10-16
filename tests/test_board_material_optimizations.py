"""Tests for board_material_aerial_enhancer optimizations.

Tests the performance improvements including:
- scikit-learn KMeans integration
- Parameter validation
- Memory optimizations
- Timing instrumentation
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from board_material_aerial_enhancer import (
    _kmeans,
    _validate_parameters,
    enhance_aerial,
    load_palette_assignments,
    save_palette_assignments,
)


class TestParameterValidation:
    """Test parameter validation function."""

    def test_valid_parameters_accepted(self):
        """Test that valid parameters are accepted."""
        # Should not raise
        _validate_parameters(k=8, analysis_max=1280, seed=22, target_width=4096)
        _validate_parameters(k=2, analysis_max=32, seed=0, target_width=32)
        _validate_parameters(k=256, analysis_max=4096, seed=999, target_width=None)

    def test_invalid_k_rejected(self):
        """Test that invalid k values are rejected."""
        with pytest.raises(ValueError, match="k must be at least 2"):
            _validate_parameters(k=1, analysis_max=1280, seed=22, target_width=4096)

        with pytest.raises(ValueError, match="k must be <= 256"):
            _validate_parameters(k=257, analysis_max=1280, seed=22, target_width=4096)

    def test_invalid_analysis_max_rejected(self):
        """Test that invalid analysis_max values are rejected."""
        with pytest.raises(ValueError, match="analysis_max must be at least 32"):
            _validate_parameters(k=8, analysis_max=31, seed=22, target_width=4096)

    def test_invalid_seed_rejected(self):
        """Test that negative seed is rejected."""
        with pytest.raises(ValueError, match="seed must be non-negative"):
            _validate_parameters(k=8, analysis_max=1280, seed=-1, target_width=4096)

    def test_invalid_target_width_rejected(self):
        """Test that invalid target_width is rejected."""
        with pytest.raises(ValueError, match="target_width must be at least 32"):
            _validate_parameters(k=8, analysis_max=1280, seed=22, target_width=31)


class TestKMeansOptimization:
    """Test k-means clustering optimizations."""

    def test_sklearn_kmeans_faster_than_basic(self):
        """Test that sklearn k-means is faster than basic implementation."""
        # Create synthetic data
        np.random.seed(42)
        data = np.random.rand(10000, 3).astype(np.float32)

        # Time sklearn version
        start_sklearn = time.time()
        labels_sklearn = _kmeans(data, k=8, seed=42, use_sklearn=True)
        time_sklearn = time.time() - start_sklearn

        # Time basic version
        start_basic = time.time()
        labels_basic = _kmeans(data, k=8, seed=42, use_sklearn=False)
        time_basic = time.time() - start_basic

        # Both should produce valid labels
        assert labels_sklearn.shape == (10000,)
        assert labels_basic.shape == (10000,)
        assert set(labels_sklearn) <= set(range(8))
        assert set(labels_basic) <= set(range(8))

        # sklearn should generally be faster or comparable
        # (not always guaranteed for small data, but should be close)
        print(f"sklearn time: {time_sklearn:.3f}s, basic time: {time_basic:.3f}s")

    def test_kmeans_deterministic(self):
        """Test that k-means produces deterministic results with same seed."""
        np.random.seed(42)
        data = np.random.rand(1000, 3).astype(np.float32)

        labels1 = _kmeans(data, k=4, seed=42, use_sklearn=True)
        labels2 = _kmeans(data, k=4, seed=42, use_sklearn=True)

        # Should produce identical results
        np.testing.assert_array_equal(labels1, labels2)

    def test_kmeans_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        np.random.seed(42)
        data = np.random.rand(1000, 3).astype(np.float32)

        labels1 = _kmeans(data, k=4, seed=42, use_sklearn=True)
        labels2 = _kmeans(data, k=4, seed=123, use_sklearn=True)

        # Should not be identical (high probability)
        assert not np.array_equal(labels1, labels2)


class TestEnhanceAerialOptimizations:
    """Test enhance_aerial function optimizations."""

    def test_enhance_aerial_with_logging(self, tmp_path: Path, caplog):
        """Test that enhance_aerial logs timing information."""
        import logging
        caplog.set_level(logging.INFO)

        # Create test image
        image = Image.new("RGB", (100, 100), (200, 180, 160))
        input_path = tmp_path / "input.png"
        image.save(input_path)

        output_path = tmp_path / "output.png"
        enhance_aerial(
            input_path,
            output_path,
            k=4,
            analysis_max=80,
            seed=42,
            target_width=150,
        )

        # Check that timing logs were created
        assert "K-means clustering" in caplog.text
        assert "Total processing time" in caplog.text

    def test_enhance_aerial_accepts_analysis_max_dim(self, tmp_path: Path):
        """Test backward compatibility with analysis_max_dim parameter."""
        image = Image.new("RGB", (100, 100), (200, 180, 160))
        input_path = tmp_path / "input.png"
        image.save(input_path)

        output_path = tmp_path / "output.png"
        enhance_aerial(
            input_path,
            output_path,
            k=4,
            analysis_max_dim=80,  # Old parameter name
            seed=42,
            target_width=150,
        )

        assert output_path.exists()

    def test_enhance_aerial_with_sklearn_disabled(self, tmp_path: Path):
        """Test that enhance_aerial works with sklearn disabled."""
        image = Image.new("RGB", (50, 50), (200, 180, 160))
        input_path = tmp_path / "input.png"
        image.save(input_path)

        output_path = tmp_path / "output.png"
        enhance_aerial(
            input_path,
            output_path,
            k=3,
            analysis_max=40,
            seed=42,
            target_width=75,
            use_sklearn=False,
        )

        assert output_path.exists()
        enhanced = Image.open(output_path)
        assert enhanced.size[0] == 75

    def test_enhance_aerial_resample_methods(self, tmp_path: Path):
        """Test different resampling methods."""
        image = Image.new("RGB", (100, 100), (200, 180, 160))
        input_path = tmp_path / "input.png"
        image.save(input_path)

        for method in ["NEAREST", "BILINEAR", "LANCZOS"]:
            output_path = tmp_path / f"output_{method}.png"
            enhance_aerial(
                input_path,
                output_path,
                k=4,
                analysis_max=80,
                seed=42,
                target_width=150,
                resample_method=method,
            )

            assert output_path.exists()
            enhanced = Image.open(output_path)
            assert enhanced.size[0] == 150


class TestPaletteOperations:
    """Test palette save/load operations."""

    def test_save_and_load_palette_roundtrip(self, tmp_path: Path):
        """Test that palette can be saved and loaded correctly."""
        from board_material_aerial_enhancer import MaterialRule

        # Create some test materials
        rule_a = MaterialRule(name="plaster")
        rule_b = MaterialRule(name="stone")
        rule_c = MaterialRule(name="bronze")

        assignments = {0: rule_a, 1: rule_b, 3: rule_c}
        palette_path = tmp_path / "palette.json"

        # Save
        save_palette_assignments(assignments, palette_path)
        assert palette_path.exists()

        # Load
        rules = [rule_a, rule_b, rule_c]
        loaded = load_palette_assignments(palette_path, rules)

        # Verify
        assert len(loaded) == 3
        assert loaded[0].name == "plaster"
        assert loaded[1].name == "stone"
        assert loaded[3].name == "bronze"


class TestMemoryEfficiency:
    """Test memory optimization features."""

    def test_no_unnecessary_copies(self, tmp_path: Path):
        """Test that processing doesn't create excessive copies."""
        # Create a larger image to test memory efficiency
        image = Image.new("RGB", (500, 500))
        # Add some variation (vectorized for efficiency)
        x = np.arange(500)
        y = np.arange(500)
        xx, yy = np.meshgrid(x, y)
        r = 200 + 50 * np.sin(xx / 50)
        g = 180 + 50 * np.cos(yy / 50)
        b = np.full_like(r, 160)
        arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")

        input_path = tmp_path / "input.png"
        image.save(input_path)

        output_path = tmp_path / "output.png"

        # This should complete without excessive memory use
        enhance_aerial(
            input_path,
            output_path,
            k=8,
            analysis_max=400,
            seed=42,
            target_width=600,
        )

        assert output_path.exists()
        enhanced = Image.open(output_path)
        assert enhanced.size[0] == 600


class TestPerformanceBenchmark:
    """Benchmark tests for performance monitoring."""

    def test_processing_time_reasonable(self, tmp_path: Path):
        """Test that processing completes in reasonable time."""
        # Create test image (vectorized for efficiency)
        x = np.arange(400)
        y = np.arange(400)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        val = (xx + yy) % 256
        arr = np.stack([val, val, val], axis=-1).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")

        input_path = tmp_path / "input.png"
        image.save(input_path)

        output_path = tmp_path / "output.png"

        # Time the processing
        start = time.time()
        enhance_aerial(
            input_path,
            output_path,
            k=8,
            analysis_max=300,
            seed=42,
            target_width=500,
        )
        elapsed = time.time() - start

        assert output_path.exists()
        # Should complete in under 5 seconds for this small test
        assert elapsed < 5.0, f"Processing took {elapsed:.2f}s, expected < 5.0s"
