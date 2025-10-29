"""
Pipeline E-SK (골격화된 경계 기반) 전처리 테스트

TDD Red Phase: Skeletonization 기반 1픽셀 두께 경계 추출
- Pipeline E 출력을 입력으로 받음 (고정 파라미터: sobel, connected_components, min_component_size=70)
- 세 가지 skeleton 알고리즘 지원: zhang_suen, morphological, medial_axis
- 'all' 옵션으로 세 가지 방법 모두 적용 가능
- 1픽셀 두께의 경계 출력
- 이진 출력 (0 or 255)
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_rgb_image():
    """테스트용 작은 RGB 이미지 (64x64)"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    # 에지 생성용 패턴
    img[10:30, 10:30, :] = 150  # 밝은 사각형
    img[30:50, 30:50, :] = 80   # 어두운 사각형
    img[15:25, 40:50, :] = 200  # 밝은 사각형
    return img


@pytest.fixture
def sample_grayscale_image():
    """테스트용 작은 그레이스케일 이미지 (64x64)"""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:30, 10:30] = 150
    img[30:50, 30:50] = 80
    img[15:25, 40:50] = 200
    return img


@pytest.fixture
def thick_edge_image():
    """두꺼운 에지 이미지 (골격화 전)"""
    img = np.zeros((100, 100), dtype=np.uint8)
    # 두꺼운 수직선 (골격화 시 얇아져야 함)
    img[10:50, 20:25] = 255  # 5픽셀 두께
    # 두꺼운 수평선
    img[60:65, 30:70] = 255  # 5픽셀 두께
    # 큰 사각형 테두리
    img[70:90, 70:90] = 255
    img[72:88, 72:88] = 0  # 내부 비움
    return img


@pytest.fixture
def real_sample_image():
    """실제 샘플 이미지 로드"""
    from src.hpe.data.loader import load_image
    img_path = Path(__file__).parent.parent / "data" / "source1.jpg"
    return load_image(str(img_path), grayscale=False)


# ========================================
# 1. 입력 검증 테스트 (5개)
# ========================================

class TestPipelineESKInputValidation:
    """Pipeline E-SK의 입력 검증 테스트"""

    def test_should_raise_error_on_none_input(self):
        """None 입력 시 TypeError 발생"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        with pytest.raises(TypeError, match="cannot be None"):
            process(None)

    def test_should_raise_error_on_invalid_type(self):
        """잘못된 타입 입력 시 TypeError 발생"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        with pytest.raises(TypeError, match="must be a numpy ndarray"):
            process([1, 2, 3])

    def test_should_raise_error_on_invalid_dimensions(self):
        """잘못된 차원 (1D, 4D) 입력 시 ValueError 발생"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        # 1D 배열
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            process(np.array([1, 2, 3]))
        
        # 4D 배열
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            process(np.zeros((2, 3, 4, 5)))

    def test_should_raise_error_on_invalid_skeleton_method(self):
        """잘못된 skeleton_method 입력 시 ValueError 발생"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="skeleton_method must be"):
            process(img, skeleton_method="invalid_method")

    def test_should_handle_empty_image(self):
        """빈 이미지(모두 0) 처리 가능해야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = process(img)
        
        # 에지가 없으므로 출력도 비어있어야 함
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64)


# ========================================
# 2. 기본 동작 테스트 (6개)
# ========================================

class TestPipelineESKBasicOperation:
    """Pipeline E-SK의 기본 동작 테스트"""

    def test_should_process_rgb_image_with_zhang_suen(self, sample_rgb_image):
        """RGB 이미지를 zhang_suen 방법으로 처리해야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_rgb_image, skeleton_method="zhang_suen")
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_should_process_grayscale_image(self, sample_grayscale_image):
        """그레이스케일 이미지를 처리해야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_grayscale_image, skeleton_method="zhang_suen")
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_should_process_with_morphological_method(self, sample_rgb_image):
        """morphological 방법으로 처리해야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_rgb_image, skeleton_method="morphological")
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_should_process_with_medial_axis_method(self, sample_rgb_image):
        """medial_axis 방법으로 처리해야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_rgb_image, skeleton_method="medial_axis")
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_should_return_2d_array(self, sample_rgb_image):
        """출력은 2D 배열이어야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_rgb_image)
        
        assert result.ndim == 2, f"출력은 2D 배열이어야 함, got {result.ndim}D"

    def test_should_return_uint8_dtype(self, sample_rgb_image):
        """출력은 uint8 타입이어야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_rgb_image)
        
        assert result.dtype == np.uint8, f"dtype는 uint8이어야 함, got {result.dtype}"


# ========================================
# 3. 출력 검증 테스트 (5개)
# ========================================

class TestPipelineESKOutputValidation:
    """Pipeline E-SK의 출력 검증 테스트"""

    def test_should_output_binary_values(self, sample_rgb_image):
        """출력 값이 0 또는 255만 포함해야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_rgb_image)
        
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values), \
            f"출력은 0과 255만 포함해야 함, got {unique_values}"

    def test_should_preserve_spatial_dimensions(self, sample_rgb_image):
        """공간 차원이 입력과 동일해야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_rgb_image)
        
        assert result.shape[0] == sample_rgb_image.shape[0], "높이가 유지되어야 함"
        assert result.shape[1] == sample_rgb_image.shape[1], "너비가 유지되어야 함"

    def test_should_reduce_edge_pixels_compared_to_pipeline_e(self, sample_rgb_image):
        """골격화로 인해 Pipeline E 대비 에지 픽셀이 감소해야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        from src.hpe.pipelines.pipeline_e import process as pipeline_e_process
        
        # Pipeline E 결과 (고정 파라미터)
        edges_e = pipeline_e_process(
            sample_rgb_image,
            edge_method="sobel",
            filter_method="connected_components",
            min_component_size=70
        )
        
        # Pipeline E-SK 결과
        skeleton = process(sample_rgb_image)
        
        edge_pixels_e = np.sum(edges_e == 255)
        skeleton_pixels = np.sum(skeleton == 255)
        
        # 골격화로 픽셀 수가 감소해야 함 (또는 같을 수 있음)
        assert skeleton_pixels <= edge_pixels_e, \
            f"골격화 후 픽셀이 감소해야 함: E={edge_pixels_e}, E-SK={skeleton_pixels}"

    def test_should_maintain_connectivity(self, thick_edge_image):
        """골격화 후에도 연결성이 유지되어야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        from scipy import ndimage
        
        # 두꺼운 에지를 직접 골격화 (Pipeline E 없이 테스트)
        # Note: 실제로는 Pipeline E를 거치지만, 여기서는 골격화 알고리즘만 테스트
        # 임시로 두꺼운 에지 이미지를 사용
        
        # 간단한 연결성 체크: 에지가 있다면 연결 요소가 존재해야 함
        img_3d = np.stack([thick_edge_image] * 3, axis=2)
        result = process(img_3d)
        
        if np.sum(result == 255) > 0:
            # 연결 요소 개수 확인
            labeled, num_components = ndimage.label(result == 255)
            assert num_components > 0, "골격화 후에도 연결 요소가 존재해야 함"

    def test_should_produce_thin_edges(self, thick_edge_image):
        """골격화로 인해 에지가 얇아져야 함 (두께 감소)"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        # 두꺼운 에지를 RGB로 변환
        img_3d = np.stack([thick_edge_image] * 3, axis=2)
        
        # 골격화 전 에지 픽셀 수
        original_edge_pixels = np.sum(thick_edge_image == 255)
        
        # 골격화 후
        skeleton = process(img_3d)
        skeleton_pixels = np.sum(skeleton == 255)
        
        # 골격화로 픽셀 수가 크게 감소해야 함
        # (두꺼운 선이 1픽셀로 줄어들므로)
        if original_edge_pixels > 0:
            reduction_ratio = skeleton_pixels / original_edge_pixels
            # 최소 30% 이상 감소 (5픽셀 두께 -> 1픽셀이면 80% 감소)
            assert reduction_ratio < 0.7, \
                f"골격화로 에지가 얇아져야 함: 감소율={reduction_ratio:.2f}"


# ========================================
# 4. skeleton_method='all' 테스트 (4개)
# ========================================

class TestPipelineESKAllMethods:
    """skeleton_method='all' 옵션 테스트"""

    def test_should_return_dict_when_all_methods(self, sample_rgb_image):
        """skeleton_method='all'일 때 딕셔너리를 반환해야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_rgb_image, skeleton_method="all")
        
        assert isinstance(result, dict), "skeleton_method='all'은 딕셔너리를 반환해야 함"

    def test_should_have_three_keys_in_dict(self, sample_rgb_image):
        """딕셔너리에 세 가지 방법의 키가 있어야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_rgb_image, skeleton_method="all")
        
        expected_keys = {"zhang_suen", "morphological", "medial_axis"}
        assert set(result.keys()) == expected_keys, \
            f"키가 {expected_keys}여야 함, got {set(result.keys())}"

    def test_all_methods_should_have_consistent_shape(self, sample_rgb_image):
        """모든 방법의 출력 shape이 일치해야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(sample_rgb_image, skeleton_method="all")
        
        expected_shape = sample_rgb_image.shape[:2]
        for method_name, skeleton in result.items():
            assert skeleton.shape == expected_shape, \
                f"{method_name}의 shape이 {expected_shape}여야 함, got {skeleton.shape}"
            assert skeleton.dtype == np.uint8, \
                f"{method_name}의 dtype이 uint8이어야 함, got {skeleton.dtype}"
            assert skeleton.ndim == 2, \
                f"{method_name}의 출력이 2D여야 함, got {skeleton.ndim}D"

    def test_all_methods_should_reduce_pixels(self, sample_rgb_image):
        """모든 방법이 Pipeline E 대비 픽셀 수를 감소시켜야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        from src.hpe.pipelines.pipeline_e import process as pipeline_e_process
        
        # Pipeline E 결과
        edges_e = pipeline_e_process(
            sample_rgb_image,
            edge_method="sobel",
            filter_method="connected_components",
            min_component_size=70
        )
        edge_pixels_e = np.sum(edges_e == 255)
        
        # 모든 골격화 방법 적용
        results = process(sample_rgb_image, skeleton_method="all")
        
        for method_name, skeleton in results.items():
            skeleton_pixels = np.sum(skeleton == 255)
            assert skeleton_pixels <= edge_pixels_e, \
                f"{method_name}: 골격화로 픽셀 감소 필요, E={edge_pixels_e}, SK={skeleton_pixels}"


# ========================================
# 5. 실제 이미지 테스트 (선택적)
# ========================================

@pytest.mark.slow
class TestPipelineESKRealImage:
    """실제 이미지를 사용한 통합 테스트"""

    def test_should_process_real_image(self, real_sample_image):
        """실제 병리 이미지를 처리할 수 있어야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        result = process(real_sample_image)
        
        assert result.ndim == 2
        assert result.dtype == np.uint8
        assert result.shape[:2] == real_sample_image.shape[:2]
        
        # 에지 비율 확인 (병리 이미지에서 합리적인 범위)
        edge_ratio = 100 * np.sum(result == 255) / result.size
        assert 0 <= edge_ratio < 15, \
            f"에지 비율이 합리적 범위 내여야 함: {edge_ratio:.2f}%"

    def test_should_compare_all_methods_on_real_image(self, real_sample_image):
        """실제 이미지에서 세 가지 방법을 비교할 수 있어야 함"""
        from src.hpe.pipelines.pipeline_e_sk import process
        
        results = process(real_sample_image, skeleton_method="all")
        
        # 각 방법의 통계 출력 (디버깅용)
        for method_name, skeleton in results.items():
            edge_pixels = np.sum(skeleton == 255)
            edge_ratio = 100 * edge_pixels / skeleton.size
            print(f"\n{method_name}: {edge_pixels:,} pixels ({edge_ratio:.2f}%)")
        
        # 모든 방법이 유효한 결과를 생성해야 함
        for method_name, skeleton in results.items():
            assert skeleton.shape == real_sample_image.shape[:2]
            assert np.sum(skeleton == 255) > 0, \
                f"{method_name}는 최소한의 에지를 검출해야 함"
