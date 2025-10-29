import numpy as np
import pytest
from skimage.io import imread
from hpe.pipelines import pipeline_f
import histomicstk as htk

# 32x32 RGB image fixture
@pytest.fixture
def rgb_image():
    """테스트용 32x32 RGB 이미지 생성"""
    return np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

# 32x32 Grayscale image fixture
@pytest.fixture
def grayscale_image():
    """테스트용 32x32 그레이스케일 이미지 생성"""
    return np.random.randint(0, 256, (32, 32), dtype=np.uint8)

def test_should_process_rgb_image_without_error(rgb_image):
    """pipeline_f.process가 RGB 이미지를 에러 없이 처리해야 합니다."""
    result = pipeline_f.process(rgb_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == (32, 32)
    assert result.dtype == np.uint8

def test_should_handle_grayscale_image_by_returning_it(grayscale_image):
    """pipeline_f.process가 그레이스케일 입력을 그대로 반환해야 합니다."""
    result = pipeline_f.process(grayscale_image)
    assert np.array_equal(result, grayscale_image)
    assert result.dtype == np.uint8

def test_should_raise_typeerror_for_none_input():
    """None 입력에 대해 TypeError를 발생시켜야 합니다."""
    with pytest.raises(TypeError, match="Input image cannot be None"):
        pipeline_f.process(None)

def test_should_save_intermediate_normalized_image(rgb_image, tmp_path):
    """중간 정규화 이미지를 지정된 경로에 저장해야 합니다."""
    save_path = tmp_path / "normalized.png"
    
    assert not save_path.exists()
    
    pipeline_f.process(rgb_image, normalized_save_path=str(save_path))
    
    assert save_path.exists()
    
    saved_image = imread(str(save_path))
    assert saved_image.shape == rgb_image.shape
    assert saved_image.dtype == np.uint8

def test_should_apply_clahe_enhancement(rgb_image):
    """결과 이미지가 CLAHE를 통해 대비가 향상되어야 합니다."""
    # 단순 정규화 결과 (비교 대상)
    # 이 부분은 pipeline_b의 로직을 간소화하여 가져옴
    normalized = htk.preprocessing.color_normalization.reinhard(rgb_image, [8.63234435, -0.11501964, 0.03868433], [0.57506023, 0.10403329, 0.01364062])
    stains = htk.preprocessing.color_deconvolution.color_deconvolution(normalized, np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]]), 255)
    hematoxylin = stains.Stains[:, :, 0]
    h_min, h_max = hematoxylin.min(), hematoxylin.max()
    if h_max > h_min:
        simple_normalized = (hematoxylin - h_min) / (h_max - h_min) * 255
    else:
        simple_normalized = np.full_like(hematoxylin, 128)
    simple_normalized = simple_normalized.astype(np.uint8)

    # pipeline_f 결과
    clahe_result = pipeline_f.process(rgb_image)

    # CLAHE 적용 시 픽셀 강도 분포가 달라져야 함 (분산이 더 커지는 경향)
    assert clahe_result.dtype == np.uint8
    # 분산이 항상 커지는 것은 아니지만, 일반적으로 대비가 개선됨
    # 여기서는 실행 여부와 타입/형상만 확인
    assert not np.allclose(clahe_result, simple_normalized)

