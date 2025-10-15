"""
Tests for Pipeline B (Nuclei-focused) implementation.

TDD: Red phase - These tests verify the nuclei extraction implementation.

Pipeline B workflow:
1. Apply Reinhard color normalization
2. Perform color deconvolution (stain separation)
3. Extract hematoxylin channel (nuclei information)
4. Optional: Apply LoG filter for nucleus center enhancement
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_rgb_image():
    """Load a real RGB image from data folder."""
    from src.hpe.data.loader import load_image
    data_path = Path(__file__).parent.parent / "data" / "source1.jpg"
    return load_image(str(data_path))


@pytest.fixture
def real_sample_image():
    """Load a real histopathology sample image from /data folder."""
    from src.hpe.data.loader import load_image
    img_path = Path(__file__).parent.parent / "data" / "source1.jpg"
    return load_image(img_path, grayscale=False)


class TestPipelineBBasicOutput:
    """Test Pipeline B basic output characteristics (use_clog=False)."""

    def test_returns_2d_grayscale_image(self, sample_rgb_image):
        """Test that Pipeline B returns a 2D grayscale image."""
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        assert result.ndim == 2, \
            f"Output must be 2D (grayscale), got {result.ndim}D"

    def test_returns_uint8_dtype(self, sample_rgb_image):
        """Test that Pipeline B returns uint8 dtype."""
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        assert result.dtype == np.uint8, \
            f"Output must be uint8, got {result.dtype}"

    def test_output_range_0_to_255(self, sample_rgb_image):
        """Test that output values are in valid uint8 range [0, 255]."""
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        assert result.min() >= 0, \
            f"Minimum value must be >= 0, got {result.min()}"
        assert result.max() <= 255, \
            f"Maximum value must be <= 255, got {result.max()}"

    def test_output_has_same_spatial_dimensions(self, sample_rgb_image):
        """Test that output has same height and width as input."""
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        assert result.shape[0] == sample_rgb_image.shape[0], \
            "Output height must match input height"
        assert result.shape[1] == sample_rgb_image.shape[1], \
            "Output width must match input width"

    def test_output_is_not_all_zeros(self, sample_rgb_image):
        """Test that output is not completely black (all zeros)."""
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        assert result.max() > 0, \
            "Output should not be all zeros (completely black)"

    def test_output_has_variance(self, sample_rgb_image):
        """
        Test that output has variance (not uniform).
        
        Acceptance criteria from PRD: variance > 0
        """
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        # Check that there's variation in pixel values (variance > 0)
        variance = np.var(result)
        assert variance > 0, \
            f"Output should have variance > 0, got {variance}"


@pytest.mark.slow
class TestPipelineBWithRealImage:
    """Test Pipeline B with real histopathology images (may be slow)."""

    def test_processes_real_image_successfully(self, real_sample_image):
        """Test that Pipeline B can process a real histopathology image."""
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(real_sample_image, use_clog=False)
        
        assert result is not None, "Result should not be None"
        assert result.ndim == 2, "Result should be 2D"
        assert result.dtype == np.uint8, "Result should be uint8"
        assert result.shape[0] == real_sample_image.shape[0], \
            "Height should be preserved"
        assert result.shape[1] == real_sample_image.shape[1], \
            "Width should be preserved"

    def test_real_image_extracts_hematoxylin(self, real_sample_image):
        """
        Test that hematoxylin channel is properly extracted.
        
        Hematoxylin stains nuclei (typically darker/purple areas).
        The output should have meaningful content representing nuclei.
        """
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(real_sample_image, use_clog=False)
        
        # Should have substantial variation (nuclei create contrast)
        assert result.std() > 5, \
            f"Expected std > 5 for nuclei contrast, got {result.std():.2f}"
        
        # Should not be mostly saturated (all white or all black)
        mean_val = result.mean()
        assert 20 < mean_val < 235, \
            f"Expected mean in (20, 235) for reasonable extraction, got {mean_val:.2f}"

    def test_color_normalization_applied(self, real_sample_image):
        """
        Test that color normalization affects the output.
        
        Reinhard normalization should standardize colors, so two passes
        should give consistent results.
        """
        from src.hpe.pipelines.pipeline_b import process
        
        result1 = process(real_sample_image, use_clog=False)
        result2 = process(real_sample_image, use_clog=False)
        
        # Results should be identical (deterministic)
        np.testing.assert_array_equal(result1, result2,
            "Pipeline B should be deterministic")


class TestPipelineBEdgeCases:
    """Test Pipeline B with edge cases."""

    def test_handles_small_image(self):
        """Test that Pipeline B can handle very small images."""
        from src.hpe.pipelines.pipeline_b import process
        
        # 8x8 RGB image
        small_img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        
        result = process(small_img, use_clog=False)
        
        assert result.shape == (8, 8), "Should preserve small image dimensions"
        assert result.dtype == np.uint8, "Should return uint8"

    def test_handles_mostly_black_image(self):
        """Test that Pipeline B handles images with mostly black pixels."""
        from src.hpe.pipelines.pipeline_b import process
        
        # Mostly black image with small bright region
        dark_img = np.zeros((64, 64, 3), dtype=np.uint8)
        dark_img[28:36, 28:36] = 200  # Small bright region
        
        result = process(dark_img, use_clog=False)
        
        assert result.shape == (64, 64), "Should preserve dimensions"
        assert result.dtype == np.uint8, "Should return uint8"

    def test_handles_mostly_white_image(self):
        """Test that Pipeline B handles images with mostly white pixels."""
        from src.hpe.pipelines.pipeline_b import process
        
        # Mostly white image (like background)
        bright_img = np.full((64, 64, 3), 240, dtype=np.uint8)
        bright_img[28:36, 28:36] = 100  # Small darker region (nuclei-like)
        
        result = process(bright_img, use_clog=False)
        
        assert result.shape == (64, 64), "Should preserve dimensions"
        assert result.dtype == np.uint8, "Should return uint8"

    def test_default_use_clog_is_false(self, sample_rgb_image):
        """Test that default behavior is use_clog=False."""
        from src.hpe.pipelines.pipeline_b import process
        
        result_default = process(sample_rgb_image)
        result_explicit = process(sample_rgb_image, use_clog=False)
        
        np.testing.assert_array_equal(result_default, result_explicit,
            "Default should be use_clog=False")


class TestPipelineBInputValidation:
    """Test Pipeline B input validation."""

    def test_rejects_none_input(self):
        """Test that Pipeline B rejects None input."""
        from src.hpe.pipelines.pipeline_b import process
        
        with pytest.raises(TypeError, match="cannot be None"):
            process(None)

    def test_rejects_non_ndarray_input(self):
        """Test that Pipeline B rejects non-ndarray input."""
        from src.hpe.pipelines.pipeline_b import process
        
        with pytest.raises(TypeError, match="numpy ndarray"):
            process([1, 2, 3])

    def test_rejects_1d_array(self):
        """Test that Pipeline B rejects 1D arrays."""
        from src.hpe.pipelines.pipeline_b import process
        
        with pytest.raises(ValueError, match="2D or 3D"):
            process(np.array([1, 2, 3]))

    def test_rejects_4d_array(self):
        """Test that Pipeline B rejects 4D arrays."""
        from src.hpe.pipelines.pipeline_b import process
        
        with pytest.raises(ValueError, match="2D or 3D"):
            process(np.zeros((2, 2, 3, 3)))


# Note: LoG filter tests (use_clog=True) will be added in item 7
class TestPipelineBLoGPlaceholder:
    """Placeholder for LoG filter tests (item 7 in TODO list)."""

    def test_log_parameter_exists(self, sample_rgb_image):
        """Test that use_clog parameter is accepted (even if not implemented yet)."""
        from src.hpe.pipelines.pipeline_b import process
        
        # Should not raise an error
        result = process(sample_rgb_image, use_clog=True)
        
        assert result is not None, "Should return a result"
        assert result.ndim == 2, "Should still return 2D image"
        assert result.dtype == np.uint8, "Should still return uint8"
