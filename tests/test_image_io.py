"""
Tests for image I/O functionality (loading images from disk).

TDD: Red phase - These tests will fail until we implement the image loader.
"""

import pytest
import numpy as np
from pathlib import Path


# Test fixture for sample image paths
@pytest.fixture
def sample_image_path():
    """Returns path to a valid sample JPG image in /data folder."""
    return Path(__file__).parent.parent / "data" / "source1.jpg"


@pytest.fixture
def another_sample_image_path():
    """Returns path to another valid sample JPG image."""
    return Path(__file__).parent.parent / "data" / "source2.jpg"


@pytest.fixture
def nonexistent_image_path():
    """Returns path to a file that doesn't exist."""
    return Path(__file__).parent.parent / "data" / "nonexistent_image.jpg"


class TestImageLoadingPositive:
    """Positive test cases: valid image loading scenarios."""

    def test_load_image_returns_ndarray(self, sample_image_path):
        """Test that loading a valid image returns a numpy ndarray."""
        from src.hpe.data.loader import load_image
        
        result = load_image(sample_image_path)
        
        assert isinstance(result, np.ndarray), \
            f"Expected np.ndarray, got {type(result)}"

    def test_load_image_has_valid_dtype(self, sample_image_path):
        """Test that loaded image has uint8 or float32 dtype."""
        from src.hpe.data.loader import load_image
        
        result = load_image(sample_image_path)
        
        assert result.dtype in [np.uint8, np.float32], \
            f"Expected dtype uint8 or float32, got {result.dtype}"

    def test_load_image_color_has_three_channels(self, sample_image_path):
        """Test that loading in color mode returns 3-channel image."""
        from src.hpe.data.loader import load_image
        
        result = load_image(sample_image_path, grayscale=False)
        
        assert result.ndim == 3, \
            f"Expected 3D array for color image, got {result.ndim}D"
        assert result.shape[2] == 3, \
            f"Expected 3 channels (RGB), got {result.shape[2]}"

    def test_load_image_grayscale_has_two_dimensions(self, sample_image_path):
        """Test that loading in grayscale mode returns 2D image."""
        from src.hpe.data.loader import load_image
        
        result = load_image(sample_image_path, grayscale=True)
        
        assert result.ndim == 2, \
            f"Expected 2D array for grayscale image, got {result.ndim}D"

    def test_load_image_has_positive_dimensions(self, sample_image_path):
        """Test that loaded image has positive width and height."""
        from src.hpe.data.loader import load_image
        
        result = load_image(sample_image_path)
        
        assert result.shape[0] > 0, "Image height must be positive"
        assert result.shape[1] > 0, "Image width must be positive"

    def test_load_image_uint8_in_valid_range(self, sample_image_path):
        """Test that uint8 image values are in [0, 255] range."""
        from src.hpe.data.loader import load_image
        
        result = load_image(sample_image_path)
        
        if result.dtype == np.uint8:
            assert result.min() >= 0, "uint8 image values must be >= 0"
            assert result.max() <= 255, "uint8 image values must be <= 255"

    def test_load_different_images_return_different_data(
        self, sample_image_path, another_sample_image_path
    ):
        """Test that loading different images returns different data."""
        from src.hpe.data.loader import load_image
        
        img1 = load_image(sample_image_path)
        img2 = load_image(another_sample_image_path)
        
        # Images should not be identical (at least one pixel differs)
        assert not np.array_equal(img1, img2), \
            "Different images should not have identical pixel values"


class TestImageLoadingNegative:
    """Negative test cases: error handling scenarios."""

    def test_load_nonexistent_file_raises_error(self, nonexistent_image_path):
        """Test that loading a non-existent file raises FileNotFoundError."""
        from src.hpe.data.loader import load_image
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_image(nonexistent_image_path)
        
        assert "does not exist" in str(exc_info.value).lower() or \
               "no such file" in str(exc_info.value).lower(), \
            "Error message should indicate file doesn't exist"

    def test_load_none_path_raises_error(self):
        """Test that passing None as path raises appropriate error."""
        from src.hpe.data.loader import load_image
        
        with pytest.raises((TypeError, ValueError)):
            load_image(None)

    def test_load_empty_string_path_raises_error(self):
        """Test that passing empty string as path raises appropriate error."""
        from src.hpe.data.loader import load_image
        
        with pytest.raises((FileNotFoundError, ValueError)):
            load_image("")

    def test_load_directory_instead_of_file_raises_error(self):
        """Test that passing a directory path raises appropriate error."""
        from src.hpe.data.loader import load_image
        
        dir_path = Path(__file__).parent.parent / "data"
        
        with pytest.raises((ValueError, OSError, FileNotFoundError)):
            load_image(dir_path)


class TestImageLoadingEdgeCases:
    """Edge cases and additional validation."""

    def test_load_image_accepts_string_path(self, sample_image_path):
        """Test that load_image accepts string paths (not just Path objects)."""
        from src.hpe.data.loader import load_image
        
        result = load_image(str(sample_image_path))
        
        assert isinstance(result, np.ndarray), \
            "Should accept string paths and return ndarray"

    def test_load_image_default_is_color(self, sample_image_path):
        """Test that default loading mode is color (not grayscale)."""
        from src.hpe.data.loader import load_image
        
        result = load_image(sample_image_path)
        
        # Default should be color (3D with 3 channels)
        assert result.ndim == 3, \
            "Default mode should load color images (3D array)"
