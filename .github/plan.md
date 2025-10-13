
# Histo-Eval - 개발 작업 및 테스트 TODO 리스트

이 파일은 `.github/tdd.md`의 규칙을 따르는 TDD 우선 순위의 작업/테스트 체크리스트입니다. 각 항목은 "작은 단위"로 나누어져 있으며, 하나씩 테스트를 작성하고 통과시키는 방식으로 진행합니다.

규칙 요약

- "Red → Green → Refactor" 사이클을 준수합니다.
- 각 항목은 먼저 실패하는 테스트를 추가한 뒤, 최소 구현으로 테스트를 통과시키고(Green), 필요한 경우 리팩터링(Refactor)합니다.
- 항목을 완료하면 체크박스를 체크합니다.

우선순위(Phase 1)

1. [x] BOOTSTRAP: 프로젝트 초기화 파일 존재 확인
   - 테스트 파일: `tests/test_bootstrap_files.py`
   - 테스트: `requirements.txt`, `config.yaml`(템플릿), `.github/prd.md`, `.github/tdd.md`, `.github/instruction.md` 파일 존재성 검사.
   - 수락 기준: 누락 시 명확한 어설션 실패 메시지.

2. [x] DATA LOADER: 이미지 쌍 목록 파싱 (CSV)
   - 테스트 파일: `tests/test_data_loader_csv.py`
   - 테스트(긍정): 샘플 CSV를 읽어 `[{image_pair_id, source_path, target_path}, ...]` 리스트 반환.
   - 테스트(부정): 필수 컬럼 누락/빈 값/잘못된 구분자에 대해 `ValueError` 발생.
   - 수락 기준: 스키마 위반 시 일관된 예외 메시지.

3. [x] DATA LOADER: 이미지 로드(파일 존재/형식 검증)
   - 테스트 파일: `tests/test_image_io.py`
   - 테스트(긍정): `/data` 폴더의 샘플 JPG 이미지를 로드하여 `np.ndarray` 반환, `dtype`는 `uint8` 또는 `float32`.
   - 테스트(부정): 존재하지 않는 경로/손상 파일에서 `FileNotFoundError`/`ValueError`.
   - 수락 기준: 로딩 옵션(그레이스케일/컬러) 플래그에 따른 채널 수 일치.

4. [x] PIPELINE: 파이프라인 모듈 인터페이스
   - 테스트: 각 파이프라인 모듈(`pipeline_a`, `pipeline_b`, `pipeline_c`, `pipeline_d`)이 `process(image: np.ndarray) -> np.ndarray` 인터페이스를 준수하는지 확인.
   - 수락 기준: 잘못된 입력(예: None, wrong dtype)에 대해 ValueError 또는 TypeError 발생.

5. [x] PIPELINE A: Baseline++ 기본 동작
   - 테스트: 작은 RGB 샘플 이미지에 대해 `pipeline_a`가 마스크 적용 후 CLAHE된 2D uint8 이미지를 반환.
   - 수락 기준: 반환 이미지가 2D이고 값 범위가 0-255.
   - 구현: `histomicstk.saliency.tissue_detection.get_tissue_mask` 사용하여 tissue mask 생성 후 CLAHE 적용.

6. [ ] PIPELINE B: 핵 중심 채널 추출 (기본)
   - 테스트: `pipeline_b(..., use_clog=False)`가 헤마톡실린 채널을 2D uint8로 반환.
   - 수락 기준: 반환 이미지의 통계(분산>0) 확인.

7. [ ] PIPELINE B: 핵 중심 채널 추출 (LoG 옵션)
   - 테스트: `pipeline_b(..., use_clog=True)`가 정상적으로 실행되고 값 범위가 0-255.

8. [ ] PIPELINE C: 에오신 채널 추출
   - 테스트: `pipeline_c`가 에오신 채널을 2D uint8로 반환.

9. [ ] PIPELINE D: 경계 기반 출력
   - 테스트: `pipeline_d`가 에지 이미지를 2D uint8(0 또는 255)로 반환.

10. [ ] METRICS: Feature Potential Score 계산기
   - 테스트 파일: `tests/test_metrics_feature_potential.py`
   - 테스트: grayscale 입력에서 코너/LoG 기반 스코어를 스칼라(float)로 반환.
   - 수락 기준: 균일 이미지에서는 0 또는 매우 작은 값, 텍스처 이미지에서는 더 큰 값.

11. [ ] METRICS: Image Entropy
   - 테스트 파일: `tests/test_metrics_entropy.py`
   - 테스트: 0~255 범위 이미지에서 엔트로피가 0 이상, 균일 이미지의 엔트로피 < 텍스처 이미지 엔트로피.

12. [ ] METRICS: Image Contrast (RMS)
   - 테스트 파일: `tests/test_metrics_contrast.py`
   - 테스트: 균일 이미지 대비≈0, 가장자리/체커보드 이미지 대비>0.

13. [ ] RUNNER: `run_evaluation.py` 기본 실행 흐름 (dry-run)
	- 테스트: 구성 파일(`config.yaml` 템플릿)을 읽고, 데이터 로더 및 선택된 파이프라인을 로드하는 'dry-run'이 성공.
	- 수락 기준: 파일 생성 없이 설정 파싱 및 파이프라인 인스턴스화 성공.

14. [ ] OUTPUT: 전처리 결과 저장(단일 쌍)
	- 테스트: 파이프라인 결과를 지정된 출력 폴더에 이미지 파일로 저장.
	- 수락 기준: 파일이 생성되고 읽을 수 있음.

15. [ ] REPORT: 결과 CSV 작성
	- 테스트: 결과 줄이 포함된 CSV(칼럼: image_pair_id, pipeline_name, output_path_source, output_path_target, feature_potential_score, image_entropy, image_contrast, processing_time)를 생성.
	- 수락 기준: CSV가 생성되고 각 칼럼이 올바른 타입/형식을 가짐.

우선순위(Phase 2)

16. [ ] PIPELINE C/D: 성능 개선 및 에지케이스 테스트
	- 테스트: 다양한 크기/밝기 이미지에 대해 안정적 동작.

17. [ ] PARALLEL: 멀티프로세싱 실험 실행(옵션)
	- 테스트: `multiprocessing`을 이용한 간단한 배치 처리 구현 테스트.

18. [ ] DOCS: `README.md` 초안 작성 및 실행 예시 추가
	- 테스트: README에 설치 및 실행 예시(위의 `pytest` 포함)가 포함되어 있음.

기타(작업 관리)

- 각 항목은 실제로 구현되기 전에 테스트를 작성해야 합니다.
- 큰 작업은 여러 개의 작은 테스트 항목으로 분해하세요.
- 변경은 가능한 한 작은 커밋 단위로 진행하고, 커밋 메시지에 구조/행동 유형을 명확히 표기하세요.

비고

- `histomicstk` 등 무거운 의존이 필요한 테스트는 최소 입력 크기와 `@pytest.mark.slow`를 고려하세요.
- 이미지 비교는 픽셀 완전일치보다 통계/범위 기반 어서션을 우선 사용하십시오.
