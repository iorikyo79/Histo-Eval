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
        # Pipeline B가 2차원(그레이스케일) 이미지를 반환하는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        assert result.ndim == 2, \
            f"Output must be 2D (grayscale), got {result.ndim}D"

    def test_returns_uint8_dtype(self, sample_rgb_image):
        # 반환 이미지의 dtype이 uint8인지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        assert result.dtype == np.uint8, \
            f"Output must be uint8, got {result.dtype}"

    def test_output_range_0_to_255(self, sample_rgb_image):
        # 출력 값이 0~255 범위에 있는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        assert result.min() >= 0, \
            f"Minimum value must be >= 0, got {result.min()}"
        assert result.max() <= 255, \
            f"Maximum value must be <= 255, got {result.max()}"

    def test_output_has_same_spatial_dimensions(self, sample_rgb_image):
        # 입력 이미지와 출력 이미지의 크기가 같은지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        assert result.shape[0] == sample_rgb_image.shape[0], \
            "Output height must match input height"
        assert result.shape[1] == sample_rgb_image.shape[1], \
            "Output width must match input width"

    def test_output_is_not_all_zeros(self, sample_rgb_image):
        # 출력 이미지가 완전히 검은색(0)으로만 이루어져 있지 않은지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=False)
        
        assert result.max() > 0, \
            "Output should not be all zeros (completely black)"

    def test_output_has_variance(self, sample_rgb_image):
        # 출력 이미지에 분산(값의 변화)이 존재하는지 확인 (PRD 기준: 분산 > 0)
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
        # 실제 조직 이미지를 정상적으로 처리하는지 확인
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
        # 헤마톡실린 채널(핵 정보)이 제대로 추출되는지 확인
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
        # 색상 정규화(Reinhard)가 결과에 영향을 주는지 확인 (두 번 실행 시 결과 동일)
        from src.hpe.pipelines.pipeline_b import process
        
        result1 = process(real_sample_image, use_clog=False)
        result2 = process(real_sample_image, use_clog=False)
        
        # Results should be identical (deterministic)
        np.testing.assert_array_equal(result1, result2,
            "Pipeline B should be deterministic")


class TestPipelineBEdgeCases:
    """Test Pipeline B with edge cases."""

    def test_handles_small_image(self):
        # 아주 작은 이미지도 정상적으로 처리하는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        # 8x8 RGB image
        small_img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        
        result = process(small_img, use_clog=False)
        
        assert result.shape == (8, 8), "Should preserve small image dimensions"
        assert result.dtype == np.uint8, "Should return uint8"

    def test_handles_mostly_black_image(self):
        # 대부분 검은색(0) 픽셀로 이루어진 이미지를 처리할 수 있는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        # Mostly black image with small bright region
        dark_img = np.zeros((64, 64, 3), dtype=np.uint8)
        dark_img[28:36, 28:36] = 200  # Small bright region
        
        result = process(dark_img, use_clog=False)
        
        assert result.shape == (64, 64), "Should preserve dimensions"
        assert result.dtype == np.uint8, "Should return uint8"

    def test_handles_mostly_white_image(self):
        # 대부분 흰색(255) 픽셀로 이루어진 이미지를 처리할 수 있는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        # Mostly white image (like background)
        bright_img = np.full((64, 64, 3), 240, dtype=np.uint8)
        bright_img[28:36, 28:36] = 100  # Small darker region (nuclei-like)
        
        result = process(bright_img, use_clog=False)
        
        assert result.shape == (64, 64), "Should preserve dimensions"
        assert result.dtype == np.uint8, "Should return uint8"

    def test_default_use_clog_is_false(self, sample_rgb_image):
        # use_clog 파라미터의 기본값이 False임을 확인
        from src.hpe.pipelines.pipeline_b import process
        
        result_default = process(sample_rgb_image)
        result_explicit = process(sample_rgb_image, use_clog=False)
        
        np.testing.assert_array_equal(result_default, result_explicit,
            "Default should be use_clog=False")


class TestPipelineBInputValidation:
    """Test Pipeline B input validation."""

    def test_rejects_none_input(self):
        # 입력값이 None일 때 예외(TypeError)가 발생하는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        with pytest.raises(TypeError, match="cannot be None"):
            process(None)

    def test_rejects_non_ndarray_input(self):
        # 입력값이 numpy 배열이 아닐 때 예외(TypeError)가 발생하는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        with pytest.raises(TypeError, match="numpy ndarray"):
            process([1, 2, 3])

    def test_rejects_1d_array(self):
        # 1차원 배열 입력 시 예외(ValueError)가 발생하는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        with pytest.raises(ValueError, match="2D or 3D"):
            process(np.array([1, 2, 3]))

    def test_rejects_4d_array(self):
        # 4차원 배열 입력 시 예외(ValueError)가 발생하는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        with pytest.raises(ValueError, match="2D or 3D"):
            process(np.zeros((2, 2, 3, 3)))


# LoG filter tests (item 7)
class TestPipelineBLoGFilter:
    """Test Pipeline B with LoG (Laplacian of Gaussian) filter enabled."""

    def test_log_parameter_exists(self, sample_rgb_image):
        # use_clog 파라미터가 존재하고 에러 없이 동작하는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        # Should not raise an error
        result = process(sample_rgb_image, use_clog=True)
        
        assert result is not None, "Should return a result"
        assert result.ndim == 2, "Should still return 2D image"
        assert result.dtype == np.uint8, "Should still return uint8"

    def test_log_returns_valid_output(self, sample_rgb_image):
        # LoG 필터 적용 시 정상적인 출력(2D uint8, 0-255 범위)을 반환하는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=True)
        
        assert result.ndim == 2, "LoG output should be 2D"
        assert result.dtype == np.uint8, "LoG output should be uint8"
        assert result.min() >= 0, "Min value should be >= 0"
        assert result.max() <= 255, "Max value should be <= 255"

    def test_log_enhances_nucleus_centers(self, sample_rgb_image):
        # LoG 필터가 핵 중심을 강조하는지 확인 (분산이나 특징 값이 증가해야 함)
        from src.hpe.pipelines.pipeline_b import process
        
        result_without_log = process(sample_rgb_image, use_clog=False)
        result_with_log = process(sample_rgb_image, use_clog=True)
        
        # LoG should enhance features (typically increases variation in certain regions)
        # The outputs should be different
        assert not np.array_equal(result_without_log, result_with_log), \
            "LoG filter should produce different output from basic processing"

    def test_log_preserves_dimensions(self, sample_rgb_image):
        # LoG 필터 적용 후에도 이미지 크기가 유지되는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(sample_rgb_image, use_clog=True)
        
        assert result.shape[0] == sample_rgb_image.shape[0], \
            "LoG output height should match input"
        assert result.shape[1] == sample_rgb_image.shape[1], \
            "LoG output width should match input"

    def test_log_handles_edge_cases(self):
        # LoG 필터가 작은 이미지나 극단적인 경우도 처리할 수 있는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        # Small image
        small_img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        result = process(small_img, use_clog=True)
        
        assert result.shape == (16, 16), "Should handle small images"
        assert result.dtype == np.uint8, "Should return uint8"

    @pytest.mark.slow
    def test_log_on_real_image(self, real_sample_image):
        # 실제 조직 이미지에 LoG 필터를 적용했을 때 정상 동작하는지 확인
        from src.hpe.pipelines.pipeline_b import process
        
        result = process(real_sample_image, use_clog=True)
        
        assert result.ndim == 2, "Should return 2D image"
        assert result.dtype == np.uint8, "Should return uint8"
        assert result.std() > 0, "Should have variation"
        # LoG typically produces more localized features
        assert result.mean() < 255, "Should not be all white"
        assert result.max() > 0, "Should not be all black"
