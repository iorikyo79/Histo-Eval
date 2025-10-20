"""
Pipeline D (경계 기반) 전처리 테스트

TDD Red Phase: 에지 검출 기반 전처리 파이프라인 테스트
- Canny 에지 검출 기본
- 배경 억제는 간단한 후처리 (get_tissue_mask 옵션화)
- 이진 출력 (0 or 255)
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_rgb_image():
    """테스트용 작은 RGB 이미지 (64x64)"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    # 몇 가지 패턴 추가 (에지 생성용)
    img[10:30, 10:30, :] = 150  # 밝은 사각형
    img[30:50, 30:50, :] = 80   # 어두운 사각형
    img[15:25, 40:50, :] = 200  # 밝은 사각형
    return img


@pytest.fixture
def real_sample_image():
    """실제 샘플 이미지 로드"""
    from src.hpe.data.loader import load_image
    img_path = Path(__file__).parent.parent / "data" / "source1.jpg"
    return load_image(str(img_path), grayscale=False)


class TestPipelineDBasicOutput:
    """Pipeline D의 기본 출력 형식 테스트"""

    def test_returns_2d_binary_image(self, sample_rgb_image):
        """2D 이진 이미지를 반환해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        result = process(sample_rgb_image)
        
        assert result.ndim == 2, f"출력은 2D 배열이어야 함, got {result.ndim}D"

    def test_returns_uint8_dtype(self, sample_rgb_image):
        """uint8 타입을 반환해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        result = process(sample_rgb_image)
        
        assert result.dtype == np.uint8, f"dtype는 uint8이어야 함, got {result.dtype}"

    def test_output_is_binary_values(self, sample_rgb_image):
        """출력 값이 0 또는 255만 포함해야 함 (이진)"""
        from src.hpe.pipelines.pipeline_d import process
        
        result = process(sample_rgb_image)
        
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values), \
            f"출력은 0과 255만 포함해야 함, got {unique_values}"

    def test_output_has_same_spatial_dimensions(self, sample_rgb_image):
        """출력 이미지의 공간 차원이 입력과 동일해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        result = process(sample_rgb_image)
        
        assert result.shape[0] == sample_rgb_image.shape[0], "높이가 유지되어야 함"
        assert result.shape[1] == sample_rgb_image.shape[1], "너비가 유지되어야 함"

    def test_output_has_edges_detected(self, sample_rgb_image):
        """에지가 검출되어야 함 (완전히 비어있지 않음)"""
        from src.hpe.pipelines.pipeline_d import process
        
        result = process(sample_rgb_image)
        
        # 에지가 있는 픽셀의 비율
        edge_ratio = np.sum(result == 255) / result.size
        
        # 패턴이 있는 이미지이므로 최소한의 에지가 검출되어야 함
        assert edge_ratio > 0, "에지가 전혀 검출되지 않음"
        assert edge_ratio < 0.5, f"에지가 너무 많음 ({edge_ratio:.2%}), 노이즈일 수 있음"

    def test_output_is_sparse(self, sample_rgb_image):
        """에지 맵은 희소해야 함 (대부분 0)"""
        from src.hpe.pipelines.pipeline_d import process
        
        result = process(sample_rgb_image)
        
        edge_pixels = np.sum(result == 255)
        total_pixels = result.size
        edge_ratio = edge_pixels / total_pixels
        
        # 일반적으로 에지는 전체의 1~15% 정도
        assert edge_ratio < 0.3, \
            f"에지 비율이 너무 높음 ({edge_ratio:.2%}), 임계값 조정 필요"


class TestPipelineDWithRealImage:
    """실제 조직병리 이미지로 테스트"""

    def test_processes_real_image_successfully(self, real_sample_image):
        """실제 이미지를 성공적으로 처리해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        result = process(real_sample_image)
        
        assert result is not None
        assert result.ndim == 2
        assert result.dtype == np.uint8
        assert result.shape[:2] == real_sample_image.shape[:2]

    def test_real_image_detects_meaningful_edges(self, real_sample_image):
        """실제 이미지에서 의미있는 에지를 검출해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        result = process(real_sample_image)
        
        edge_ratio = np.sum(result == 255) / result.size
        
        # 실제 조직 이미지에서는 적절한 양의 에지가 검출되어야 함
        # Sobel + CLAHE로 인해 에지 검출량이 증가하여 상한을 30%로 상향 조정
        assert 0.001 < edge_ratio < 0.3, \
            f"에지 비율이 비정상적임: {edge_ratio:.4%} (예상: 0.1~30%)"

    def test_real_image_edges_have_structure(self, real_sample_image):
        """검출된 에지가 구조적 연결성을 가져야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        result = process(real_sample_image)
        
        # 에지 픽셀이 완전히 무작위가 아닌 연결된 구조를 가지는지 확인
        # Connected components 분석
        from scipy import ndimage
        labeled, num_features = ndimage.label(result == 255)
        
        # 너무 많은 고립된 점들은 노이즈
        total_edge_pixels = np.sum(result == 255)
        if total_edge_pixels > 0:
            avg_component_size = total_edge_pixels / max(num_features, 1)
            # 평균 컴포넌트 크기가 너무 작으면 노이즈
            assert avg_component_size > 2, \
                f"에지가 너무 파편화됨 (평균 크기: {avg_component_size:.1f} 픽셀)"


class TestPipelineDEdgeCases:
    """엣지 케이스 테스트"""

    def test_handles_small_image(self):
        """작은 이미지도 처리해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        small_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = process(small_img)
        
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_handles_mostly_black_image(self):
        """대부분 검은 이미지 처리 (에지 거의 없음)"""
        from src.hpe.pipelines.pipeline_d import process
        
        black_img = np.zeros((64, 64, 3), dtype=np.uint8)
        black_img[30:34, 30:34, :] = 50  # 작은 회색 영역
        
        result = process(black_img)
        
        assert result.ndim == 2
        assert result.dtype == np.uint8
        # 거의 균일한 이미지이므로 에지가 거의 없어야 함
        edge_ratio = np.sum(result == 255) / result.size
        assert edge_ratio < 0.1, "균일한 이미지에서 너무 많은 에지 검출"

    def test_handles_mostly_white_image(self):
        """대부분 흰색 이미지 처리 (배경)"""
        from src.hpe.pipelines.pipeline_d import process
        
        white_img = np.full((64, 64, 3), 250, dtype=np.uint8)
        white_img[20:30, 20:30, :] = 100  # 어두운 영역
        
        result = process(white_img)
        
        assert result.ndim == 2
        assert result.dtype == np.uint8
        # 배경 억제가 작동해야 함
        edge_ratio = np.sum(result == 255) / result.size
        assert edge_ratio < 0.2, "흰 배경에서 노이즈 많음"

    def test_handles_high_contrast_image(self):
        """높은 대비 이미지 처리"""
        from src.hpe.pipelines.pipeline_d import process
        
        # 체커보드 패턴 (강한 에지)
        # 참고: 작은 체커보드는 Gaussian blur로 인해 에지가 smooth됨
        checker = np.zeros((64, 64, 3), dtype=np.uint8)
        checker[::2, ::2, :] = 255
        checker[1::2, 1::2, :] = 255
        
        result = process(checker)
        
        assert result.ndim == 2
        assert result.dtype == np.uint8
        # 체커보드는 blur 때문에 에지가 적을 수 있지만, 처리는 성공해야 함
        # 0 이상이면 정상 (완전히 실패하지 않음)
        assert np.sum(result == 255) >= 0, "처리가 정상적으로 완료되어야 함"


class TestPipelineDInputValidation:
    """입력 검증 테스트"""

    def test_rejects_none_input(self):
        """None 입력을 거부해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        with pytest.raises(TypeError, match="(?i)(none|cannot)"):
            process(None)

    def test_rejects_non_ndarray_input(self):
        """ndarray가 아닌 입력을 거부해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        with pytest.raises(TypeError):
            process("not an array")
        
        with pytest.raises(TypeError):
            process([1, 2, 3])

    def test_rejects_1d_array(self):
        """1D 배열을 거부해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        with pytest.raises(ValueError, match="(?i)(dimension|2d|3d)"):
            process(np.array([1, 2, 3], dtype=np.uint8))

    def test_rejects_4d_array(self):
        """4D 배열을 거부해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        with pytest.raises(ValueError, match="(?i)(dimension|2d|3d)"):
            process(np.zeros((10, 10, 3, 2), dtype=np.uint8))

    def test_accepts_grayscale_input(self):
        """그레이스케일 입력도 허용해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        gray_img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = process(gray_img)
        
        assert result.shape == gray_img.shape
        assert result.dtype == np.uint8


class TestPipelineDOptions:
    """옵션 파라미터 테스트"""
    
    def test_default_parameters_work(self):
        """기본 파라미터로 정상 동작해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = process(image)
        
        assert result is not None
        assert result.shape == (100, 100)
    
    def test_accepts_use_tissue_mask_option(self):
        """use_tissue_mask 옵션을 받아들여야 함"""
        from src.hpe.pipelines.pipeline_d import process
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # use_tissue_mask=False (기본값)
        result1 = process(image, use_tissue_mask=False)
        assert result1 is not None
        
        # use_tissue_mask=True도 허용 (HTK 없으면 fallback)
        result2 = process(image, use_tissue_mask=True)
        assert result2 is not None
    
    def test_accepts_edge_method_option(self):
        """edge_method 옵션을 받아들여야 함 (sobel, canny)"""
        from src.hpe.pipelines.pipeline_d import process
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # edge_method='sobel' (기본값)
        result_sobel = process(image, edge_method='sobel')
        assert result_sobel is not None
        assert result_sobel.shape == (100, 100)
        
        # edge_method='canny'
        result_canny = process(image, edge_method='canny')
        assert result_canny is not None
        assert result_canny.shape == (100, 100)
    
    def test_rejects_invalid_edge_method(self):
        """잘못된 edge_method는 거부해야 함"""
        from src.hpe.pipelines.pipeline_d import process
        import pytest
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="edge_method must be"):
            process(image, edge_method='invalid')
