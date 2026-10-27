# etch-pipeline

식각 공정 AI 대리 모델의 검증·재학습 자동화 파이프라인(데이터 생성 → 학습 → 평가 → 회귀 게이트 → 서빙 → 재학습). 정답 데이터는 ViennaPS 시뮬레이션이다. 팹에서 수집한 데이터는 쓰지 않는다.

상세 사양은 `docs/project-definition.md`(무엇을 왜), `docs/project-design.md`(컴포넌트·API), `docs/project-checklist.md`(주차별 작업)를 참조한다.

현재 1주차(스켈레톤) 단계이며 설치·실행 가능한 데모는 아직 없다. 진행 상황은 `docs/project-checklist.md`의 체크박스와 `reports/evidence/`를 기준으로 판단한다.

## 라이선스

저장소 코드는 Apache-2.0 (`LICENSE`). ViennaPS(GPL-3.0)를 포함하는 시뮬레이션 이미지는 별도 조건이 있다 — `NOTICE` 참조.
