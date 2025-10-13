
# Histo-Eval - 개발자 안내 (Instruction)

이 문서는 Histo-Eval 프로젝트를 시작할 때 개발자가 따라야 할 절차와 규칙을 정리합니다. 프로젝트의 기능적/비기능적 요구사항은 `.github/prd.md`에 정의되어 있으며, 코드 작성은 TDD(테스트 주도 개발) 원칙을 엄격히 따릅니다. TDD 규칙은 `.github/tdd.md`에 기술되어 있고, 실제 작업할 TODO 목록은 `.github/plan.md`에 있습니다.

주요 원칙 요약

- 항상 `.github/plan.md`의 다음 미표시(체크되지 않은) 테스트/작업을 하나씩 수행합니다.
- 각 변경은 TDD 사이클(Red → Green → Refactor)을 따릅니다: 먼저 실패하는 테스트를 추가(또는 활성화), 최소 구현으로 테스트 통과, 리팩터링(테스트 통과 확인).
- 구조 변경(리팩터링)과 행위 변경(기능 추가)은 별도의 커밋으로 나눕니다.

테스트 작성 규칙(중요)

- 테스트 파일은 `tests/` 하위에 `test_*.py`로 작성합니다.
- 테스트 함수명은 동작을 설명하는 문장형으로 작성합니다. 예: `test_should_parse_csv_pairs`.
- 각 테스트는 2초 내로 끝나는 빠른 단위 테스트여야 합니다. 더 오래 걸리면 `@pytest.mark.slow`로 표시하고 기본 실행에서 제외합니다.
- 파일 I/O가 필요한 경우 `tmp_path`/`tmp_path_factory` 픽스처를 사용합니다. 테스트 데이터는 `tests/fixtures/`에 소형 샘플로 둡니다.
- 외부 네트워크/클라우드 의존은 금지합니다. 난수 사용 시 시드 고정으로 결정적 결과를 보장합니다.
- 무거운 라이브러리 호출은 최소 이미지(예: 32×32, 64×64)로 단위 기능만 검증하세요.

필요한 파일과 위치

- 요구사항(제품 문서): `.github/prd.md`
- TDD 규칙: `.github/tdd.md`
- 작업/테스트 TODO 리스트: `.github/plan.md`
- 실행 스크립트 엔트리포인트(예시): `run_evaluation.py` (프로젝트 루트 또는 `src/`)
- 설정: `config.yaml`
- 의존성: `requirements.txt`

프로젝트 폴더 구조 제안

```
Histo-Eval/
	src/hpe/               # 모듈 코드 (data_loader, pipelines, metrics, runner 등)
		__init__.py
		data/
		pipelines/
		metrics/
	tests/                 # 단위 테스트
		fixtures/
	.github/
		prd.md
		tdd.md
		plan.md
		instruction.md
	run_evaluation.py      # 실행 스크립트(필요 시)
	config.yaml            # 설정(샘플 템플릿)
	requirements.txt
	README.md
```

개발 환경 세팅 (필수)

1. Anaconda 가상환경 생성 및 활성화

```bash
conda create -n histo-eval python=3.10 -y
conda activate histo-eval
pip install --upgrade pip
pip install -r requirements.txt
```

**중요**: 모든 작업은 반드시 `histo-eval` 가상환경 내에서 수행해야 합니다.

2. 테스트 실행

```bash
pytest -q
```

선택: 느린 테스트 제외/포함

```bash
pytest -q -m "not slow"     # 기본 권장
pytest -q -m slow            # 느린 테스트만 별도로
```

3. 스타일/정적분석 (선택)

```bash
flake8
mypy .
```

테스트 및 커밋 규칙

- 모든 변경은 관련 테스트 추가/수정과 함께 커밋되어야 합니다.
- 커밋 메시지는 변경 유형을 명확히 표기합니다. 예: "BEHAVIOR: add data loader for csv input" 또는 "STRUCTURAL: extract utils module".
- 커밋은 "한 논리 작업 단위"만 포함해야 합니다.

작업 워크플로

1. `.github/plan.md`에서 다음 체크박스(미표시) 항목을 찾습니다.
2. 해당 항목을 작은 단위 테스트(pytest)를 작성하여 실패 상태로 만듭니다.
3. 최소 구현 코드를 작성하여 테스트를 통과시킵니다.
4. 리팩터링이 필요하면 테스트가 통과하는 상태에서 구조적 수정을 수행하고 별도 커밋합니다.
5. 항목 완료 후 `.github/plan.md`에서 해당 항목을 체크(완료)합니다.

테스트 명명/조직 팁

- 테스트 파일은 기능 단위로 그룹화: `tests/test_data_loader_csv.py`, `tests/test_pipelines_a.py` 등
- 공통 픽스처/유틸은 `tests/conftest.py`에 배치
- 임시 파일/폴더는 테스트 종료 시 자동 정리되도록 `tmp_path` 기반으로만 생성

문제 발생 시

- 테스트나 빌드 실패가 발생하면 원인을 파악하고 작은 단위로 고칩니다. 큰 변경은 여러 단계로 나눠 커밋하세요.
- 외부 네트워크 호출이나 비밀(토큰 등)은 코드에 직접 포함하지 마세요.

기타

- PRD(.github/prd.md)를 항상 참조하여 기능 우선순위와 범위를 확인하세요.
- 추가 질문이나 제안은 이슈로 남기고, 주요 설계 변경은 사전에 팀 합의를 거치세요.

참고 링크

- 기능 요구사항/마일스톤: `.github/prd.md`
- TDD 규칙: `.github/tdd.md`
- 작업/테스트 목록: `.github/plan.md`

간단 체크리스트

- [ ] `.github/prd.md` 읽기
- [ ] `.github/tdd.md` 읽기
- [ ] `.github/plan.md`에서 첫 미표시 항목 찾기
- [ ] 개발 환경 구성 및 `pytest` 실행
