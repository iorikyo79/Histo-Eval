# Histo-Eval (HistoReg-Preproc Evaluator)

TDD-driven evaluation framework for histopathology image registration preprocessing pipelines.

- EN: Implements multiple preprocessing strategies (baseline++, nuclei/eosin-based, edge-based) using HistomicsTK/OpenCV/scikit-image; computes proxy metrics; and generates reports to compare pipeline candidates before running registration.
- KR: HistomicsTK/OpenCV/scikit-image를 활용한 전처리 파이프라인(Baseline++, 핵/에오신 기반, 경계 기반)을 구현하고, 프록시 지표를 계산하여 실제 정합 전에 유망한 후보군을 데이터 기반으로 비교/선정할 수 있게 해주는 평가 프레임워크입니다. 개발은 TDD 원칙을 따릅니다.

## Links

- Product Requirements: `.github/prd.md`
- TDD Rules: `.github/tdd.md`
- Development Plan: `.github/plan.md`
- Agent Guide: `agent.md`

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q -m "not slow"
```

## Proposed structure

```
Histo-Eval/
  src/hpe/
    pipelines/
    metrics/
    data/
  tests/
    fixtures/
  .github/
    prd.md
    tdd.md
    plan.md
    instruction.md
  run_evaluation.py
  config.yaml
```

## Topics (proposed)

- histopathology, digital-pathology, image-processing, image-registration, preprocessing
- histomicstk, opencv, scikit-image, numpy, python
- tdd, evaluation-framework, computer-vision

## Repository metadata automation (optional)

- Update `.github/repo-metadata.json` and push to main, or run the workflow manually.
- Add a repo secret `REPO_ADMIN_TOKEN` (PAT with `repo` scope) in GitHub Settings → Secrets and variables → Actions to let the workflow update description/topics.
