# HistoReg-Preproc Evaluator (HPE) PRD

**문서 버전:** 1.0  
**작성일:** 2025년 10월 13일

---

## 1. 개요 (Overview)

### 1.1. 프로젝트명
* **HPE (HistoReg-Preproc Evaluator)**: 조직 병리 이미지 정합을 위한 전처리 파이프라인 평가 프레임워크

### 1.2. 배경 및 문제 정의
현재 Registration AI 프로젝트는 흑백 변환 및 CLAHE를 사용하는 단순한 전처리 파이프라인에 의존하고 있습니다. 이 방식은 조직 병리 이미지가 가진 고유의 문제(염색 가변성, 조직 변형, 아티팩트 등)에 효과적으로 대응하지 못하며, 이는 SuperPoint 모델이 의미 없는 특징점을 추출하게 만들어 정합 성능 저하의 원인이 될 수 있습니다. 이 문제를 해결하기 위해 HistomicsTK를 활용한 여러 고급 전처리 전략이 제안되었으며, 각 전략의 잠재적 성능을 정량적이고 체계적으로 비교할 수 있는 표준화된 실험 환경이 필요합니다.

### 1.3. 목표
* 제안된 4가지 전처리 파이프라인을 모듈화된 Python 코드로 구현합니다.
* 지정된 이미지 쌍에 각 파이프라인을 적용하여, 후속 정합 모델의 입력으로 사용될 **결과 이미지를 생성하고 저장**합니다.
* 정합 성공 가능성을 예측할 수 있는 **자체 평가 지표(Proxy Metrics)를 계산**하여, 각 파이프라인의 잠재적 성능을 정량적으로 비교 분석할 수 있는 리포트를 생성합니다.
* 최종적으로 실제 정합을 수행하기 전, 가장 유망한 전처리 파이프라인 후보군을 데이터 기반으로 선정하는 것을 목표로 합니다.

### 1.4. 대상 사용자
* INFINITT Healthcare AI 개발팀

---

## 2. 기능 요구사항 (Functional Requirements)

### 2.1. 데이터 로더 (Data Loader)
* **FR-1**: 지정된 폴더 경로에서 이미지 쌍(원본, 타겟)을 불러올 수 있어야 합니다.
* **FR-2**: 실험할 이미지 쌍의 목록이 정의된 CSV 또는 JSON 파일을 입력받아 해당 데이터만 처리할 수 있어야 합니다.

### 2.2. 전처리 파이프라인 모듈 (Preprocessing Pipeline Modules)
각 파이프라인은 별도의 Python 파일로 구현되어야 하며, 공통된 `process(image)` 인터페이스를 가져야 합니다.

#### **FR-4: 파이프라인 A (Baseline++)**
* **설명**: 아티팩트에 강건한 조직 영역 마스크를 생성한 후, 기존 방식(흑백 변환 + CLAHE)을 적용합니다.
* **샘플 코드**:
    ```python
    import cv2
    import numpy as np
    import histomicstk as htk

    def pipeline_a(im_rgb: np.ndarray) -> np.ndarray:
        # 1. Get tissue mask robust to artifacts
        im_mask = htk.saliency.get_tissue_mask(
            im_rgb, deconvolve_first=True, n_thresholding_steps=1, sigma=1.5
        )[0]

        # 2. Convert to grayscale
        im_gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)

        # 3. Apply mask
        im_masked_gray = im_gray * (im_mask / 255)

        # 4. Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        im_clahe = clahe.apply(im_masked_gray.astype(np.uint8))
        
        return im_clahe
    ```

#### **FR-5: 파이프라인 B (핵 중심 정합)**
* **설명**: 색상 정규화 후 염색을 분리하여 '핵' 정보를 담고 있는 헤마톡실린 채널을 추출합니다. 선택적으로 LoG 필터를 적용해 핵 중심을 더욱 강조할 수 있습니다.
* **샘플 코드**:
    ```python
    import numpy as np
    import histomicstk as htk

    # Define a standard target for color normalization
    TARGET_MU = np.array([8.63, -0.01, 0.04])
    TARGET_SIGMA = np.array([0.59, 0.11, 0.02])

    def pipeline_b(im_rgb: np.ndarray, use_clog: bool = False) -> np.ndarray:
        # 1. Color normalization
        im_nmzd = htk.preprocessing.color_normalization.reinhard(
            im_rgb, TARGET_MU, TARGET_SIGMA
        )

        # 2. Color deconvolution
        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
        w = np.array([stain_color_map['hematoxylin'], stain_color_map['eosin'], stain_color_map['null']]).T
        im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, w).Stains

        # 3. Extract Hematoxylin channel (nuclei)
        im_hematoxylin = im_stains[:, :, 0]

        # 4. (Optional) Enhance nuclei centers with LoG filter
        if use_clog:
            im_hematoxylin = htk.filters.shape.clog(
                im_hematoxylin, sigma_min=2*np.sqrt(2), sigma_max=4*np.sqrt(2)
            )
        
        # Normalize to 0-255 for visualization/saving
        im_hematoxylin = (im_hematoxylin - np.min(im_hematoxylin)) / (np.max(im_hematoxylin) - np.min(im_hematoxylin)) * 255
        return im_hematoxylin.astype(np.uint8)
    ```

#### **FR-6: 파이프라인 C (텍스처 중심 정합)**
* **설명**: 파이프라인 B와 동일한 과정을 거치되, 세포질과 기질 정보를 담고 있는 에오신 채널을 추출합니다.
* **샘플 코드**:
    ```python
    import numpy as np
    import histomicstk as htk

    # Use the same target for color normalization
    TARGET_MU = np.array([8.63, -0.01, 0.04])
    TARGET_SIGMA = np.array([0.59, 0.11, 0.02])

    def pipeline_c(im_rgb: np.ndarray) -> np.ndarray:
        # 1. Color normalization
        im_nmzd = htk.preprocessing.color_normalization.reinhard(
            im_rgb, TARGET_MU, TARGET_SIGMA
        )

        # 2. Color deconvolution
        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
        w = np.array([stain_color_map['hematoxylin'], stain_color_map['eosin'], stain_color_map['null']]).T
        im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, w).Stains

        # 3. Extract Eosin channel (stroma/cytoplasm)
        im_eosin = im_stains[:, :, 1]

        # Normalize to 0-255
        im_eosin = (im_eosin - np.min(im_eosin)) / (np.max(im_eosin) - np.min(im_eosin)) * 255
        return im_eosin.astype(np.uint8)
    ```

#### **FR-7: 파이프라인 D (경계 기반 정합)**
* **설명**: 색상 정규화 후, 조직의 구조적 윤곽선을 강조하기 위해 경계 검출 알고리즘을 적용합니다.
* **샘플 코드**:
    ```python
    import cv2
    import numpy as np
    import histomicstk as htk
    from skimage.feature import canny # Note: Using scikit-image for Canny detector

    def pipeline_d(im_rgb: np.ndarray) -> np.ndarray:
        # 1. Get tissue mask
        im_mask = htk.saliency.get_tissue_mask(
            im_rgb, deconvolve_first=True
        )[0]
        
        # 2. Convert to grayscale
        im_gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
        im_masked_gray = (im_gray * (im_mask / 255)).astype(np.uint8)

        # 3. Apply Canny edge detector
        im_edges = canny(im_masked_gray, sigma=1.5, low_threshold=0.05, high_threshold=0.15)
        
        # Convert boolean to uint8 for saving
        return (im_edges * 255).astype(np.uint8)
    ```

### 2.3. 파이프라인 실행기 및 결과 생성기
* **FR-8**: `run_evaluation.py` 스크립트를 통해 전체 평가 프로세스를 실행합니다.
* **FR-9**: 평가할 파이프라인과 데이터셋 경로는 `config.yaml` 파일로 지정합니다.
* **FR-10**: 각 이미지 쌍에 대해 선택된 파이프라인을 적용하고, **전처리된 결과 이미지를 지정된 경로에 저장**합니다.
* **FR-11**: 전처리된 결과 이미지로부터 자체 평가 지표를 계산하고 결과를 저장합니다.

### 2.4. 평가 지표 (Proxy Metrics)
* **FR-12: 특징점 잠재력 (Feature Potential Score)**
    * LoG 필터 응답이나 FAST 코너 개수를 측정하여 이미지 내 특징점 후보의 풍부함을 정량화합니다.
* **FR-13: 이미지 엔트로피 (Image Entropy)**
    * 이미지의 정보량을 측정하여 너무 단조롭거나 노이즈가 과하지 않은지 평가합니다.
* **FR-14: 이미지 대비 (Image Contrast)**
    * RMS contrast 등을 측정하여 특징점 검출에 유리한 환경인지를 간접적으로 평가합니다.

### 2.5. 결과 리포팅
* **FR-15**: 모든 실험 결과를 단일 CSV 파일로 출력하며, 컬럼은 다음과 같습니다.
    > `image_pair_id`, `pipeline_name`, `output_path_source`, `output_path_target`, `feature_potential_score`, `image_entropy`, `image_contrast`, `processing_time`

---

## 3. 비기능 요구사항 (Non-Functional Requirements)

* **NF-1 (개발 환경)**: Python 3.9+, `histomicstk`, `opencv-python`, `numpy`, `scikit-image`, `pyyaml`. 모든 의존성은 `requirements.txt`로 관리합니다.
* **NF-2 (코드 구조)**: 각 기능(데이터 로딩, 파이프라인, 평가, 리포팅)이 디렉토리별로 명확하게 분리된 모듈형 구조를 가집니다.
* **NF-3 (설정 관리)**: 데이터 경로, 파이프라인별 파라미터 등 주요 설정값들을 `config.yaml` 파일로 관리하여 실험 재현성을 보장합니다.
* **NF-4 (문서화)**: `README.md` 파일에 프로젝트 설치, 설정, 실행 방법을 기술합니다. 모든 주요 함수에는 Docstring을 작성합니다.

---

## 4. 릴리즈 계획 (Milestones)

* **Phase 1: 코어 프레임워크 및 Baseline 구축**
    * 프로젝트 기본 구조 설정 (`config.yaml`, `requirements.txt`)
    * 데이터 로더 및 결과 CSV 저장 기능 구현
    * **파이프라인 A (Baseline++)** 및 **파이프라인 B (핵 중심)** 구현
    * 모든 Proxy Metrics (FR-12, 13, 14) 계산 모듈 구현
* **Phase 2: 전체 파이프라인 완성 및 테스트**
    * **파이프라인 C (텍스처 중심)** 및 **파이프라인 D (경계 기반)** 구현
    * 샘플 데이터셋에 대한 전체 파이프라인 실행 및 결과 리포팅 기능 검증
    * `README.md` 문서 초안 작성 및 코드 리뷰

---

## 5. 향후 계획 (Future Considerations)
* **실험 관리 도구 연동**: MLflow 또는 Weights & Biases와 연동하여 실험 결과 및 파라미터를 체계적으로 추적합니다.
* **병렬 처리**: 대규모 데이터셋 평가 속도 향상을 위해 `multiprocessing`을 이용한 병렬 처리 기능 도입을 고려합니다.