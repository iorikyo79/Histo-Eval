# Pipeline 검증 스크립트

이 폴더에는 각 파이프라인의 동작을 실제 데이터로 검증하고 결과를 시각적으로 비교하기 위한 스크립트가 포함되어 있습니다.

## 사용 가능한 스크립트

### verify_pipeline_a.py
Pipeline A (Baseline++) 검증 스크립트

**기능:**
- Tissue mask 생성 및 CLAHE 적용
- 실제 이미지 쌍에 대해 처리 수행
- 결과를 `output/pipeline_a/` 폴더에 저장

**사용법:**
```bash
# 기본 실행 (이미지 저장)
python tests/verify_pipeline_a.py

# 이미지 저장 없이 검증만
python tests/verify_pipeline_a.py --no-save
```

### verify_pipeline_b.py
Pipeline B (Nuclei-focused) 검증 스크립트

**기능:**
- Reinhard 색상 정규화
- Color deconvolution을 통한 헤마톡실린 채널 추출
- 선택적 LoG 필터 적용
- 기본 모드와 LoG 모드 결과를 별도 폴더에 저장

**사용법:**
```bash
# 기본 실행 (기본 모드 + LoG 모드)
python tests/verify_pipeline_b.py

# 기본 모드만 테스트
python tests/verify_pipeline_b.py --no-log

# 이미지 저장 없이 검증만
python tests/verify_pipeline_b.py --no-save

# LoG 없이 검증만
python tests/verify_pipeline_b.py --no-save --no-log
```

**출력 폴더:**
- `output/pipeline_b_basic/` - 기본 헤마톡실린 채널 추출 결과
- `output/pipeline_b_clog/` - LoG 필터 적용 결과 (핵 중심 강조)

## 출력 파일 구조

각 스크립트는 다음과 같은 형식으로 파일을 생성합니다:

```
output/
├── pipeline_a/
│   ├── pair_001_source_processed.png
│   ├── pair_001_target_processed.png
│   ├── pair_002_source_processed.png
│   └── ...
├── pipeline_b_basic/
│   ├── pair_001_source_basic.png
│   ├── pair_001_target_basic.png
│   └── ...
└── pipeline_b_clog/
    ├── pair_001_source_clog.png
    ├── pair_001_target_clog.png
    └── ...
```

## 검증 내용

각 스크립트는 다음 항목을 검증합니다:

1. **출력 형식**: 2D 배열, uint8 타입
2. **값 범위**: 0-255
3. **공간 차원**: 입력과 동일한 크기 유지
4. **내용**: 비어있지 않고 분산이 있는지 확인
5. **통계**: min, max, mean, std 출력

## 비교 분석

생성된 이미지들을 비교하여 다음을 확인할 수 있습니다:

### Pipeline A vs Pipeline B (basic)
- Pipeline A: CLAHE 적용으로 전반적인 대비 향상
- Pipeline B (basic): 핵 정보만 강조, 세포질 정보 제거

### Pipeline B (basic) vs Pipeline B (clog)
- basic: 헤마톡실린 채널 원본 (모든 핵 정보 포함)
- clog: LoG 필터로 핵 중심만 강조 (blob 검출)

## 예제

```bash
# Pipeline A 실행
cd /mnt/Disk1/source/Histo-Eval
python tests/verify_pipeline_a.py

# Pipeline B 두 모드 모두 실행
python tests/verify_pipeline_b.py

# 결과 이미지 확인
ls -lh output/pipeline_*/
```

## 참고

- CSV 파일: `tests/fixtures/test_pipeline_a.csv`
- 테스트용 이미지 경로는 CSV 파일에 정의되어 있음
- 각 이미지 쌍마다 source와 target 이미지 처리
