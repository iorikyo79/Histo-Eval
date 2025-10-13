"""
Tests for pipeline module interface compliance.

TDD: Red phase - These tests will fail until we implement the pipeline interface.

Each pipeline module must implement a common interface:
- process(image: np.ndarray) -> np.ndarray
- Proper error handling for invalid inputs
"""

import pytest
import numpy as np
from pathlib import Path


# Test fixture for sample image
@pytest.fixture
def sample_rgb_image():
    """Returns a small RGB test image (64x64)."""
    # Create a simple test pattern
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    # Add some pattern to make it non-uniform
    img[10:30, 10:30, 0] = 150  # Red region
    img[30:50, 30:50, 1] = 200  # Green region
    img[20:40, 20:40, 2] = 180  # Blue region
    return img


@pytest.fixture
def real_sample_image():
    """Load a real sample image from /data folder."""
    from src.hpe.data.loader import load_image
    img_path = Path(__file__).parent.parent / "data" / "source1.jpg"
    return load_image(img_path, grayscale=False)


class TestPipelineAInterface:
    """Test pipeline_a interface compliance."""

    def test_pipeline_a_exists(self):
        """Test that pipeline_a module can be imported."""
        try:
            from src.hpe.pipelines import pipeline_a
            assert hasattr(pipeline_a, 'process'), \
                "pipeline_a module must have a 'process' function"
        except ImportError as e:
            pytest.fail(f"Failed to import pipeline_a: {e}")

    def test_pipeline_a_accepts_ndarray(self, sample_rgb_image):
        """Test that pipeline_a.process accepts numpy ndarray input."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(sample_rgb_image)
        
        assert isinstance(result, np.ndarray), \
            f"process() must return np.ndarray, got {type(result)}"

    def test_pipeline_a_returns_valid_output(self, sample_rgb_image):
        """Test that pipeline_a.process returns valid image data."""
        from src.hpe.pipelines.pipeline_a import process
        
        result = process(sample_rgb_image)
        
        assert result.dtype in [np.uint8, np.float32, np.float64], \
            f"Output dtype must be numeric, got {result.dtype}"
        assert result.size > 0, "Output must not be empty"

    def test_pipeline_a_rejects_none(self):
        """Test that pipeline_a.process raises TypeError for None input."""
        from src.hpe.pipelines.pipeline_a import process
        
        with pytest.raises(TypeError) as exc_info:
            process(None)
        
        assert "none" in str(exc_info.value).lower() or \
               "invalid" in str(exc_info.value).lower(), \
            "Error message should indicate None is not accepted"

    def test_pipeline_a_rejects_wrong_dtype(self):
        """Test that pipeline_a.process raises error for non-array input."""
        from src.hpe.pipelines.pipeline_a import process
        
        with pytest.raises((TypeError, ValueError)):
            process("not an array")

    def test_pipeline_a_rejects_wrong_dimensions(self):
        """Test that pipeline_a.process raises error for 1D or 4D arrays."""
        from src.hpe.pipelines.pipeline_a import process
        
        # 1D array
        with pytest.raises((ValueError, TypeError)):
            process(np.array([1, 2, 3], dtype=np.uint8))
        
        # 4D array
        with pytest.raises((ValueError, TypeError)):
            process(np.zeros((10, 10, 3, 2), dtype=np.uint8))


class TestPipelineBInterface:
    """Test pipeline_b interface compliance."""

    def test_pipeline_b_exists(self):
        """Test that pipeline_b module can be imported."""
        try:
            from src.hpe.pipelines import pipeline_b
            assert hasattr(pipeline_b, 'process'), \
                "pipeline_b module must have a 'process' function"
        except ImportError as e:
            pytest.fail(f"Failed to import pipeline_b: {e}")

    def test_pipeline_b_accepts_ndarray(self, sample_rgb_image):
        """Test that pipeline_b.process accepts numpy ndarray input."""
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image)
        
        assert isinstance(result, np.ndarray), \
            f"process() must return np.ndarray, got {type(result)}"

    def test_pipeline_b_rejects_none(self):
        """Test that pipeline_b.process raises TypeError for None input."""
        from src.hpe.pipelines.pipeline_b import process
        
        with pytest.raises(TypeError):
            process(None)

    def test_pipeline_b_has_optional_parameters(self, sample_rgb_image):
        """Test that pipeline_b.process accepts optional use_clog parameter."""
        from src.hpe.pipelines.pipeline_b import process
        
        # Should work with default parameters
        result1 = process(sample_rgb_image)
        assert isinstance(result1, np.ndarray)
        
        # Should work with explicit use_clog=False
        result2 = process(sample_rgb_image, use_clog=False)
        assert isinstance(result2, np.ndarray)
        
        # Should work with use_clog=True
        result3 = process(sample_rgb_image, use_clog=True)
        assert isinstance(result3, np.ndarray)


class TestPipelineCInterface:
    """Test pipeline_c interface compliance."""

    def test_pipeline_c_exists(self):
        """Test that pipeline_c module can be imported."""
        try:
            from src.hpe.pipelines import pipeline_c
            assert hasattr(pipeline_c, 'process'), \
                "pipeline_c module must have a 'process' function"
        except ImportError as e:
            pytest.fail(f"Failed to import pipeline_c: {e}")

    def test_pipeline_c_accepts_ndarray(self, sample_rgb_image):
        """Test that pipeline_c.process accepts numpy ndarray input."""
        from src.hpe.pipelines.pipeline_c import process
        
        result = process(sample_rgb_image)
        
        assert isinstance(result, np.ndarray), \
            f"process() must return np.ndarray, got {type(result)}"

    def test_pipeline_c_rejects_none(self):
        """Test that pipeline_c.process raises TypeError for None input."""
        from src.hpe.pipelines.pipeline_c import process
        
        with pytest.raises(TypeError):
            process(None)


class TestPipelineDInterface:
    """Test pipeline_d interface compliance."""

    def test_pipeline_d_exists(self):
        """Test that pipeline_d module can be imported."""
        try:
            from src.hpe.pipelines import pipeline_d
            assert hasattr(pipeline_d, 'process'), \
                "pipeline_d module must have a 'process' function"
        except ImportError as e:
            pytest.fail(f"Failed to import pipeline_d: {e}")

    def test_pipeline_d_accepts_ndarray(self, sample_rgb_image):
        """Test that pipeline_d.process accepts numpy ndarray input."""
        from src.hpe.pipelines.pipeline_d import process
        
        result = process(sample_rgb_image)
        
        assert isinstance(result, np.ndarray), \
            f"process() must return np.ndarray, got {type(result)}"

    def test_pipeline_d_rejects_none(self):
        """Test that pipeline_d.process raises TypeError for None input."""
        from src.hpe.pipelines.pipeline_d import process
        
        with pytest.raises(TypeError):
            process(None)


class TestAllPipelinesConsistency:
    """Test that all pipelines follow consistent interface patterns."""

    def test_all_pipelines_importable(self):
        """Test that all four pipeline modules can be imported."""
        pipelines = ['pipeline_a', 'pipeline_b', 'pipeline_c', 'pipeline_d']
        
        for pipeline_name in pipelines:
            try:
                module = __import__(f'src.hpe.pipelines.{pipeline_name}', 
                                  fromlist=['process'])
                assert hasattr(module, 'process'), \
                    f"{pipeline_name} must have 'process' function"
            except ImportError as e:
                pytest.fail(f"Failed to import {pipeline_name}: {e}")

    def test_all_pipelines_return_ndarray(self, sample_rgb_image):
        """Test that all pipelines return numpy ndarrays."""
        from src.hpe.pipelines import pipeline_a, pipeline_b, pipeline_c, pipeline_d
        
        pipelines = [
            ('pipeline_a', pipeline_a.process),
            ('pipeline_b', pipeline_b.process),
            ('pipeline_c', pipeline_c.process),
            ('pipeline_d', pipeline_d.process),
        ]
        
        for name, process_func in pipelines:
            result = process_func(sample_rgb_image)
            assert isinstance(result, np.ndarray), \
                f"{name} must return np.ndarray, got {type(result)}"

    def test_all_pipelines_reject_none(self):
        """Test that all pipelines raise TypeError for None input."""
        from src.hpe.pipelines import pipeline_a, pipeline_b, pipeline_c, pipeline_d
        
        pipelines = [
            ('pipeline_a', pipeline_a.process),
            ('pipeline_b', pipeline_b.process),
            ('pipeline_c', pipeline_c.process),
            ('pipeline_d', pipeline_d.process),
        ]
        
        for name, process_func in pipelines:
            with pytest.raises(TypeError, match="(?i)(none|invalid|type)"):
                process_func(None)
