"""
Tests for Pipeline A (Baseline++) implementation.

TDD: Red phase - These tests verify the full Baseline++ implementation.

Pipeline A workflow:
1. Get artifact-robust tissue mask using histomicstk
2. Convert to grayscale
3. Apply mask to grayscale image
4. Apply CLAHE enhancement
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_rgb_image():
    """Load a real RGB image from data folder."""
    from src.hpe.data.loader import load_image
    data_path = Path(__file__).parent.parent / "data" / "265_HE_test.jpg"
    return load_image(str(data_path))


@pytest.fixture
def real_sample_image():
    """Load a real histopathology sample image from /data folder."""
    from src.hpe.data.loader import load_image
    img_path = Path(__file__).parent.parent / "data" / "source1.jpg"
    return load_image(img_path, grayscale=False)


class TestPipelineAOutput:
    """Test Pipeline A output characteristics."""

    def test_returns_2d_grayscale_image(self, sample_rgb_image):
        """Test that Pipeline A returns a 2D grayscale image."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(sample_rgb_image)
        
        assert result.ndim == 2, \
            f"Output must be 2D (grayscale), got {result.ndim}D"

    def test_returns_uint8_dtype(self, sample_rgb_image):
        """Test that Pipeline A returns uint8 dtype."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(sample_rgb_image)
        
        assert result.dtype == np.uint8, \
            f"Output must be uint8, got {result.dtype}"

    def test_output_range_0_to_255(self, sample_rgb_image):
        """Test that output values are in valid uint8 range [0, 255]."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(sample_rgb_image)
        
        assert result.min() >= 0, \
            f"Minimum value must be >= 0, got {result.min()}"
        assert result.max() <= 255, \
            f"Maximum value must be <= 255, got {result.max()}"

    def test_output_has_same_spatial_dimensions(self, sample_rgb_image):
        """Test that output has same height and width as input."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(sample_rgb_image)
        
        assert result.shape[0] == sample_rgb_image.shape[0], \
            "Output height must match input height"
        assert result.shape[1] == sample_rgb_image.shape[1], \
            "Output width must match input width"

    def test_output_is_not_all_zeros(self, sample_rgb_image):
        """Test that output is not completely black (all zeros)."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(sample_rgb_image)
        
        assert result.max() > 0, \
            "Output should not be all zeros (completely black)"

    def test_output_has_contrast(self, sample_rgb_image):
        """Test that output has some contrast (not uniform)."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(sample_rgb_image)
        
        # Check that there's variation in pixel values
        assert result.std() > 0, \
            "Output should have some variation (not all same value)"


@pytest.mark.slow
class TestPipelineAWithRealImage:
    """Test Pipeline A with real histopathology images (may be slow)."""

    def test_processes_real_image_successfully(self, real_sample_image):
        """Test that Pipeline A can process a real histopathology image."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(real_sample_image)
        
        assert result is not None, "Result should not be None"
        assert result.ndim == 2, "Result should be 2D"
        assert result.dtype == np.uint8, "Result should be uint8"
        assert result.shape[0] == real_sample_image.shape[0], \
            "Height should be preserved"
        assert result.shape[1] == real_sample_image.shape[1], \
            "Width should be preserved"

    def test_real_image_output_has_tissue_content(self, real_sample_image):
        """Test that processed real image contains tissue (not mostly black)."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(real_sample_image)
        
        # Real tissue images should have substantial non-zero content
        non_zero_ratio = np.count_nonzero(result) / result.size
        assert non_zero_ratio > 0.1, \
            f"Expected > 10% non-zero pixels in tissue image, got {non_zero_ratio*100:.1f}%"

    def test_real_image_output_enhanced(self, real_sample_image):
        """Test that CLAHE enhancement is applied (output has good contrast)."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(real_sample_image)
        
        # CLAHE should produce good contrast
        # Check that histogram is reasonably spread out
        hist, _ = np.histogram(result[result > 0], bins=50, range=(0, 255))
        # At least some bins should have values (not concentrated in few bins)
        non_empty_bins = np.count_nonzero(hist)
        assert non_empty_bins > 10, \
            f"Expected contrast enhancement (>10 histogram bins used), got {non_empty_bins}"


class TestPipelineAEdgeCases:
    """Test Pipeline A with edge cases."""

    def test_handles_small_image(self):
        """Test that Pipeline A can handle very small images."""
        from src.hpe.pipelines.pipeline_a import process
        
        # 8x8 RGB image
        small_img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        
        result = process(small_img)
        
        assert result.shape == (8, 8), "Should preserve small image dimensions"
        assert result.dtype == np.uint8, "Should return uint8"

    def test_handles_mostly_black_image(self):
        """Test that Pipeline A handles images with mostly black pixels."""
        from src.hpe.pipelines.pipeline_a import process
        
        # Mostly black image with small bright region
        dark_img = np.zeros((64, 64, 3), dtype=np.uint8)
        dark_img[28:36, 28:36] = 200  # Small bright region
        
        result = process(dark_img)
        
        assert result.shape == (64, 64), "Should preserve dimensions"
        assert result.dtype == np.uint8, "Should return uint8"

    def test_handles_mostly_white_image(self):
        """Test that Pipeline A handles images with mostly white pixels."""
        from src.hpe.pipelines.pipeline_a import process
        
        # Mostly white image with small dark region
        bright_img = np.full((64, 64, 3), 240, dtype=np.uint8)
        bright_img[28:36, 28:36] = 20  # Small dark region
        
        result = process(bright_img)
        
        assert result.shape == (64, 64), "Should preserve dimensions"
        assert result.dtype == np.uint8, "Should return uint8"

    def test_handles_uniform_gray_image(self):
        """Test that Pipeline A handles uniform gray images."""
        from src.hpe.pipelines.pipeline_a import process
        
        # Uniform gray image
        gray_img = np.full((64, 64, 3), 128, dtype=np.uint8)
        
        result = process(gray_img)
        
        assert result.shape == (64, 64), "Should preserve dimensions"
        assert result.dtype == np.uint8, "Should return uint8"
