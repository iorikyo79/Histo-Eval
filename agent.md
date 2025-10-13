# Repository Agent - 책임 및 작업 지침

이 저장소에서 자동화 에이전트(또는 유지보수/개발 담당자)가 따라야 할 규칙과 역할을 정리합니다. 이 파일은 사람 또는 자동화 에이전트가 코드 변경, 테스트 추가, CI 작업을 수행할 때 참조할 체크리스트입니다.

주요 역할

- TDD 사이클을 따르는 테스트/개발 작업 수행: `.github/tdd.md`의 규칙을 항상 준수합니다.
- 작업 우선순위와 구체적인 테스트는 `.github/plan.md`를 기준으로 한 단계씩 처리합니다.
- 제품 요구사항은 `.github/prd.md`를 우선 참조합니다.

작업 흐름(에이전트용 단축 버전)

1. `git pull`로 최신 코드를 가져옵니다.
2. `.github/plan.md`에서 다음 미완료(체크되지 않은) 항목을 찾습니다.
3. 해당 항목에 대한 작은 단위 테스트를 작성하여 실패 상태(Red)로 만듭니다.
4. 최소 구현을 작성하여 테스트를 통과(Green)시킵니다.
5. 필요한 경우 리팩터(Refactor)하고 테스트 재실행.
6. 변경은 구조 변경(STRUCTURAL)과 행동 변경(BEHAVIOR)으로 분리하여 커밋합니다.
7. 완료 시 `.github/plan.md`의 해당 항목 체크박스를 표시(완료)합니다.

CI 및 로컬 검증

- 로컬에서 테스트 실행:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

- 린트/타입 검사(선택): `flake8`, `mypy .`
 
마커 사용(느린 테스트):

```bash
pytest -q -m "not slow"   # 기본: 느린 테스트 제외
pytest -q -m slow          # 느린 테스트만 실행
```

문서 및 리포팅

- 새로운 기능 또는 변경은 `README.md`에 간단한 사용 예시를 추가하세요.
- 실험 구성은 `config.yaml`에 기록하고 변경 이력은 PR 설명에 포함시키세요.

문의

- 설계 변경 또는 우선순위 변경이 필요하면 이슈를 열어 팀 합의를 요청하세요.
