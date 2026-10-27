# CLAUDE.md

## 프로젝트
식각 공정 AI 대리 모델의 검증·재학습 자동화 파이프라인. 8주.
정본 문서: docs/project-definition.md, docs/project-design.md, docs/project-checklist.md
충돌 시 우선순위: definition > design > checklist. 어긋나면 임의 해결하지 말고 DOC- 항목으로 보고한다.

## 작업 규칙
1. 작업 단위는 checklist의 항목 ID(W<주>-<번호>). 한 번에 최대 2개.
2. 항목 완료 = 코드 + 테스트 + CI 통과 + 증거 한 줄(실행 명령과 출력 요약). 증거 없이 체크박스를 바꾸지 않는다.
3. 지표·게이트 코드는 테스트를 먼저 작성한다(design 6, 7장 정의 그대로).
4. 실행해 보지 않은 것은 사실로 쓰지 않는다. 문서·코드·주석에 [확인]/[제안]/[미확인] 태그를 유지한다.
5. 스파이크(S0~S6)는 시간 상자를 지키고 판정표 파일을 남긴다. 초과하면 중단하고 대안을 제시한다.
6. 인터페이스(스키마, 해시 규칙, 종료 코드)는 2주차 말에 동결. 변경은 ADR 작성 후에만.
7. 게이트(G1~G8) 판정은 근거를 보여주고 사용자 확인을 받은 뒤 다음 단계로 간다.
8. 선택 범위(definition 5.2)는 증거 ①②가 확보되기 전에 착수하지 않는다.

## 금지
- 제외 기술 재제안 금지: PhysicsNeMo Curator, ONNX/Triton, Slurm, MLflow, DVC, Transolver
- 금지 표현(코드·README·대시보드·모델 카드 전부): "대체한다", "실시간", "실제 fab 데이터", "SK하이닉스 모델 재현", "어떤 AI가 효율적인지 입증"
- 근거 없는 수치·성능·로그 작성 금지. 측정하지 않은 값은 비워 두고 [미확인]으로 표기.
- 데이터(`data/`), 키, 시크릿 커밋 금지. API 키는 환경 변수(ETCH_API_KEY)로만.
- 사용자 확인 없이 의존성 버전 변경, 라이선스 선택 변경, 외부 서비스 가입 금지.

## 환경
- 학습·추론 CPU만. ViennaPS 4.5.0 + ViennaLS 5.8.5(5.9.0 사용 금지: import 실패 [확인]).
- MeshGraphNet은 PyG 2.8.1 + torch_scatter 2.1.2 필요(`pip install --no-build-isolation torch_scatter`).
- PyPI torch는 CUDA 라이브러리를 끌어온다(~10GB). CPU 휠 인덱스 사용 여부는 S0에서 확인.
- 패키지명 `etchpipe`, 환경 3종 sim/train/serve 분리.

## 보고 형식
작업 끝에: 완료한 항목 ID, 증거 한 줄씩, 실패·미확인 사항, 다음 항목 제안. 장황한 설명 금지.