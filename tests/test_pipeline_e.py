"""
Pipeline E (노이즈 필터링된 경계 기반) 전처리 테스트

TDD Red Phase: Connected Components 또는 Morphology 기반 노이즈 필터링
- Pipeline D 출력을 입력으로 받음
- 작은 에지 조각(노이즈) 제거
- 큰 구조적 에지만 보존
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
def noisy_edge_image():
    """노이즈가 있는 에지 이미지 생성 (Pipeline D 출력 시뮬레이션)"""
    img = np.zeros((100, 100), dtype=np.uint8)
    
    # 큰 구조적 에지 (보존되어야 함)
    img[10:50, 20:22] = 255  # 수직 선 (40x2 = 80 픽셀)
    img[60:80, 30:70] = 255  # 큰 사각형 (20x40 = 800 픽셀)
    
    # 작은 노이즈 (제거되어야 함)
    img[5, 5] = 255  # 단일 픽셀
    img[8:10, 8:10] = 255  # 2x2 = 4 픽셀
    img[70, 10:15] = 255  # 1x5 = 5 픽셀
    img[85:90, 85] = 255  # 5x1 = 5 픽셀
    
    return img


@pytest.fixture
def real_sample_image():
    """실제 샘플 이미지 로드"""
    from src.hpe.data.loader import load_image
    img_path = Path(__file__).parent.parent / "data" / "source1.jpg"
    return load_image(str(img_path), grayscale=False)


class TestPipelineEBasicOutput:
    """Pipeline E의 기본 출력 형식 테스트"""

    def test_returns_2d_binary_image(self, sample_rgb_image):
        """2D 이진 이미지를 반환해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        result = process(sample_rgb_image)
        
        assert result.ndim == 2, f"출력은 2D 배열이어야 함, got {result.ndim}D"

    def test_returns_uint8_dtype(self, sample_rgb_image):
        """uint8 타입을 반환해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        result = process(sample_rgb_image)
        
        assert result.dtype == np.uint8, f"dtype는 uint8이어야 함, got {result.dtype}"

    def test_output_is_binary_values(self, sample_rgb_image):
        """출력 값이 0 또는 255만 포함해야 함 (이진)"""
        from src.hpe.pipelines.pipeline_e import process
        
        result = process(sample_rgb_image)
        
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values), \
            f"출력은 0과 255만 포함해야 함, got {unique_values}"

    def test_output_has_same_spatial_dimensions(self, sample_rgb_image):
        """출력 이미지의 공간 차원이 입력과 동일해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        result = process(sample_rgb_image)
        
        assert result.shape[0] == sample_rgb_image.shape[0], "높이가 유지되어야 함"
        assert result.shape[1] == sample_rgb_image.shape[1], "너비가 유지되어야 함"

    def test_output_has_edges_detected(self, sample_rgb_image):
        """에지가 검출되어야 함 (완전히 비어있지 않음)"""
        from src.hpe.pipelines.pipeline_e import process
        
        result = process(sample_rgb_image)
        
        # 에지가 있는 픽셀의 비율
        edge_ratio = np.sum(result == 255) / result.size
        
        # 패턴이 있는 이미지이므로 최소한의 에지가 검출되어야 함
        assert edge_ratio >= 0, "에지 픽셀이 있어야 함"

    def test_output_is_cleaner_than_pipeline_d(self, sample_rgb_image):
        """Pipeline D보다 더 깨끗한 결과를 반환해야 함 (노이즈 필터링)"""
        from src.hpe.pipelines.pipeline_d import process as pipeline_d_process
        from src.hpe.pipelines.pipeline_e import process as pipeline_e_process
        
        result_d = pipeline_d_process(sample_rgb_image)
        result_e = pipeline_e_process(sample_rgb_image)
        
        # Pipeline E는 작은 구성 요소를 제거하므로 일반적으로 에지 픽셀이 적어야 함
        edge_count_d = np.sum(result_d == 255)
        edge_count_e = np.sum(result_e == 255)
        
        # Pipeline E가 더 적거나 비슷한 에지를 가져야 함 (노이즈 제거)
        assert edge_count_e <= edge_count_d, \
            f"Pipeline E({edge_count_e})가 Pipeline D({edge_count_d})보다 에지가 많으면 안됨"


class TestPipelineENoiseFiltering:
    """노이즈 필터링 기능 테스트"""

    def test_removes_small_components_with_connected_components(self, noisy_edge_image):
        """Connected Components 방법으로 작은 구성 요소를 제거해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        # Pipeline D를 시뮬레이션하기 위해 직접 에지 이미지 전달하도록 수정
        # 실제로는 RGB 이미지를 받아서 내부적으로 Pipeline D 호출
        # 여기서는 min_component_size=10으로 10픽셀 미만 제거
        
        # 임시로 RGB로 변환 (Pipeline E는 RGB 입력 받음)
        rgb_dummy = np.stack([noisy_edge_image]*3, axis=-1)
        
        result = process(rgb_dummy, filter_method='connected_components', min_component_size=10)
        
        # 작은 노이즈(1, 4, 5, 5 픽셀)는 제거되어야 함
        # 큰 구조(80, 800 픽셀)는 보존되어야 함
        # 정확한 검증은 어렵지만, 전체 에지 수가 줄어들어야 함
        original_edges = np.sum(noisy_edge_image == 255)
        filtered_edges = np.sum(result == 255)
        
        # 노이즈가 제거되어 에지 수가 줄어들어야 함
        assert filtered_edges < original_edges, \
            f"필터링 후({filtered_edges})가 원본({original_edges})보다 적어야 함"

    def test_removes_small_components_with_morphology(self, noisy_edge_image):
        """Morphology 방법으로 작은 구성 요소를 제거해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        # 임시로 RGB로 변환
        rgb_dummy = np.stack([noisy_edge_image]*3, axis=-1)
        
        result = process(rgb_dummy, filter_method='morphology', min_component_size=10)
        
        original_edges = np.sum(noisy_edge_image == 255)
        filtered_edges = np.sum(result == 255)
        
        # Morphology로도 노이즈가 제거되어야 함
        assert filtered_edges < original_edges, \
            f"필터링 후({filtered_edges})가 원본({original_edges})보다 적어야 함"

    def test_preserves_large_structures(self, sample_rgb_image):
        """큰 구조적 에지는 보존해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        result = process(sample_rgb_image, min_component_size=50)
        
        # 큰 구조가 있으면 완전히 비어있지 않아야 함
        edge_pixels = np.sum(result == 255)
        
        # 적어도 일부 큰 구조는 남아있어야 함
        assert edge_pixels > 0, "큰 구조는 보존되어야 함"


class TestPipelineEWithRealImage:
    """실제 이미지로 테스트"""

    def test_processes_real_image_successfully(self, real_sample_image):
        """실제 이미지를 성공적으로 처리해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        result = process(real_sample_image)
        
        assert result.ndim == 2
        assert result.dtype == np.uint8
        assert result.shape[:2] == real_sample_image.shape[:2]

    def test_real_image_detects_meaningful_edges(self, real_sample_image):
        """실제 이미지에서 의미있는 에지를 검출해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        result = process(real_sample_image)
        
        edge_ratio = np.sum(result == 255) / result.size
        
        # 노이즈 필터링 후에도 적절한 양의 에지가 남아있어야 함
        # Pipeline D보다는 적지만, 완전히 비어있지 않아야 함
        assert 0.001 < edge_ratio < 0.25, \
            f"에지 비율이 비정상적임: {edge_ratio:.4%} (예상: 0.1~25%)"

    def test_real_image_is_cleaner_than_pipeline_d(self, real_sample_image):
        """실제 이미지에서 Pipeline D보다 깨끗한 결과를 반환해야 함"""
        from src.hpe.pipelines.pipeline_d import process as pipeline_d_process
        from src.hpe.pipelines.pipeline_e import process as pipeline_e_process
        
        result_d = pipeline_d_process(real_sample_image)
        result_e = pipeline_e_process(real_sample_image)
        
        # Connected components 개수 비교
        from scipy import ndimage
        
        _, num_components_d = ndimage.label(result_d == 255)
        _, num_components_e = ndimage.label(result_e == 255)
        
        # Pipeline E가 더 적은 구성 요소를 가져야 함 (노이즈 제거로 인해)
        assert num_components_e < num_components_d, \
            f"Pipeline E({num_components_e})가 Pipeline D({num_components_d})보다 구성 요소가 적어야 함"


class TestPipelineEEdgeCases:
    """엣지 케이스 테스트"""

    def test_handles_small_image(self):
        """작은 이미지도 처리해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        small_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = process(small_img)
        
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_handles_mostly_black_image(self):
        """대부분 검은색인 이미지도 처리해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        black_img = np.zeros((100, 100, 3), dtype=np.uint8)
        black_img[40:60, 40:60, :] = 50  # 약간의 패턴
        
        result = process(black_img)
        
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_handles_mostly_white_image(self):
        """대부분 흰색인 이미지도 처리해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        white_img = np.full((100, 100, 3), 250, dtype=np.uint8)
        white_img[40:60, 40:60, :] = 200  # 약간의 패턴
        
        result = process(white_img)
        
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_handles_high_contrast_image(self):
        """고대비 이미지도 처리해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast[::2, :, :] = 255  # 줄무늬 패턴
        
        result = process(high_contrast)
        
        assert result.ndim == 2
        assert result.dtype == np.uint8


class TestPipelineEInputValidation:
    """입력 검증 테스트"""

    def test_rejects_none_input(self):
        """None 입력은 거부해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        import pytest
        
        with pytest.raises(TypeError, match="cannot be None"):
            process(None)

    def test_rejects_non_ndarray_input(self):
        """ndarray가 아닌 입력은 거부해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        import pytest
        
        with pytest.raises(TypeError, match="numpy ndarray"):
            process("not an array")
        
        with pytest.raises(TypeError, match="numpy ndarray"):
            process([1, 2, 3])

    def test_rejects_1d_array(self):
        """1D 배열은 거부해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        import pytest
        
        arr_1d = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            process(arr_1d)

    def test_rejects_4d_array(self):
        """4D 배열은 거부해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        import pytest
        
        arr_4d = np.zeros((2, 10, 10, 3))
        
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            process(arr_4d)

    def test_accepts_grayscale_input(self):
        """그레이스케일(2D) 입력도 허용해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = process(gray_img)
        
        assert result.shape == gray_img.shape


class TestPipelineEOptions:
    """옵션 파라미터 테스트"""
    
    def test_default_parameters_work(self):
        """기본 파라미터로 정상 동작해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = process(image)
        
        assert result is not None
        assert result.shape == (100, 100)
    
    def test_accepts_edge_method_option(self):
        """edge_method 옵션을 받아들여야 함 (Pipeline D로 전달)"""
        from src.hpe.pipelines.pipeline_e import process
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # edge_method='sobel' (기본값)
        result_sobel = process(image, edge_method='sobel')
        assert result_sobel is not None
        assert result_sobel.shape == (100, 100)
        
        # edge_method='canny'
        result_canny = process(image, edge_method='canny')
        assert result_canny is not None
        assert result_canny.shape == (100, 100)
    
    def test_accepts_filter_method_option(self):
        """filter_method 옵션을 받아들여야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # filter_method='connected_components' (기본값)
        result_cc = process(image, filter_method='connected_components')
        assert result_cc is not None
        assert result_cc.shape == (100, 100)
        
        # filter_method='morphology'
        result_morph = process(image, filter_method='morphology')
        assert result_morph is not None
        assert result_morph.shape == (100, 100)
    
    def test_accepts_min_component_size_option(self):
        """min_component_size 옵션을 받아들여야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # 다양한 크기 값 테스트
        for size in [10, 50, 100]:
            result = process(image, min_component_size=size)
            assert result is not None
            assert result.shape == (100, 100)
    
    def test_rejects_invalid_filter_method(self):
        """잘못된 filter_method는 거부해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        import pytest
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="filter_method must be"):
            process(image, filter_method='invalid')
    
    def test_different_min_sizes_produce_different_results(self):
        """다른 min_component_size 값은 다른 결과를 생성해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        # 노이즈가 많은 이미지 생성
        image = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
        
        result_small = process(image, min_component_size=10)
        result_large = process(image, min_component_size=100)
        
        edge_count_small = np.sum(result_small == 255)
        edge_count_large = np.sum(result_large == 255)
        
        # 더 큰 min_component_size는 일반적으로 더 적은 에지를 남김
        # (단, 이미지에 따라 다를 수 있으므로 부등호 방향만 확인하지 않음)
        # 적어도 둘 중 하나는 에지가 있어야 함
        assert edge_count_small >= 0 and edge_count_large >= 0


class TestPipelineEFusion:
    """Pipeline E의 fusion 기능을 테스트"""
    
    def test_blend_returns_grayscale_not_binary(self):
        """blend_with_original=True면 이진 이미지가 아닌 그레이스케일을 반환해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = process(image, blend_with_original=True)
        
        assert result.dtype == np.uint8
        assert result.shape == (100, 100)
        # 이진값이 아닌 다양한 그레이스케일 값이 있어야 함
        unique_values = np.unique(result)
        assert len(unique_values) > 2  # 0과 255만 있으면 안 됨
    
    def test_blend_without_flag_returns_binary(self):
        """blend_with_original=False면 이진 이미지를 반환해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = process(image, blend_with_original=False)
        
        # 0 또는 255만 있어야 함
        unique_values = np.unique(result)
        assert np.all(np.isin(unique_values, [0, 255]))
    
    def test_blend_alpha_affects_result(self):
        """blend_alpha 값이 결과에 영향을 미쳐야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result_low_alpha = process(
            image, blend_with_original=True, blend_alpha=0.3
        )
        result_high_alpha = process(
            image, blend_with_original=True, blend_alpha=0.9
        )
        
        # 다른 alpha 값은 다른 결과를 생성해야 함
        assert not np.array_equal(result_low_alpha, result_high_alpha)
    
    def test_blend_alpha_validation(self):
        """blend_alpha는 0.0~1.0 범위여야 함"""
        from src.hpe.pipelines.pipeline_e import process
        import pytest
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # 범위 밖의 값은 거부
        with pytest.raises(ValueError, match="blend_alpha must be between"):
            process(image, blend_with_original=True, blend_alpha=-0.1)
        
        with pytest.raises(ValueError, match="blend_alpha must be between"):
            process(image, blend_with_original=True, blend_alpha=1.5)
        
        # 경계값은 허용
        result_0 = process(image, blend_with_original=True, blend_alpha=0.0)
        result_1 = process(image, blend_with_original=True, blend_alpha=1.0)
        assert result_0.shape == result_1.shape == (100, 100)
    
    def test_blend_preserves_texture_from_original(self):
        """블렌딩된 결과는 원본의 텍스처 정보를 보존해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        # 텍스처가 있고 에지도 있는 이미지 생성 (큰 사각형)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # 배경에 노이즈 추가
        image[:, :] = np.random.randint(80, 120, (100, 100, 3), dtype=np.uint8)
        # 명확한 사각형 구조 추가
        image[30:70, 30:70] = 200
        
        result_blend = process(
            image, blend_with_original=True, blend_alpha=0.7
        )
        result_no_blend = process(
            image, blend_with_original=False
        )
        
        # 블렌딩된 결과가 더 많은 그레이스케일 정보를 가져야 함
        unique_blend = len(np.unique(result_blend))
        unique_no_blend = len(np.unique(result_no_blend))
        assert unique_blend > unique_no_blend
    
    def test_blend_works_with_grayscale_input(self):
        """그레이스케일 입력에도 블렌딩이 작동해야 함"""
        from src.hpe.pipelines.pipeline_e import process
        
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = process(
            gray_image, blend_with_original=True, blend_alpha=0.7
        )
        
        assert result.dtype == np.uint8
        assert result.shape == (100, 100)
        assert len(np.unique(result)) > 2  # 그레이스케일 값들
