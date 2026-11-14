# W8-10: 프로젝트 전체 예외 케이스 카탈로그 (v1.0 이후 추가 정리)

8주 동안 결정·수정하면서 코드 곳곳에 흩어져 있던 "일반 규칙에서 벗어난 예외/특수
처리"를 한곳에 모은다. 새로 만든 규칙이 아니라 **이미 존재하는 코드·설정·주석을
찾아 인용**한 것이며, 숫자·근거는 전부 실제 파일 내용에서 그대로 가져왔다.

## A. 게이트(C5) 판정 로직의 예외

1. **종료 코드 2는 성능 판정을 건너뜀** — `src/etchpipe/gate/gate.py:313,316` —
   `dataset_hash`/골든 해시 불일치 시 `GateResult(passed=False, exit_code=2, ...)`를
   성능 체크 전에 바로 반환한다.
2. **신·구 게이트 함수가 같은 설정 파일을 다르게 읽음** — `configs/pipeline/gate.yaml`의
   `cd_floor`/`recipe_pass_rate`/`rollout_min_increase`/`alpha_retrain`은 "`gate.py`가
   아직 안 읽음"이라는 주석이 있으나, 이는 **구버전** `run_gate()`에만 해당 —
   ADR-0005 신버전 `check_recipe_pass_rate`/`check_no_regression`(`retrain/execute.py`의
   `make_gate_fn`이 `gate_cfg["cd_floor"]` 등을 실제로 읽음)은 이미 사용 중.
   `scripts/run_gate_negative.py`가 "기존 run_gate"/"ADR-0005 신규 경로" 둘 다 테스트.
3. **골든 Chamfer·퇴보·지연 3개는 "게이트처럼 보이지만 차단은 안 함"** — ADR-0004
   이후 ①golden Chamfer 상한·③챔피언 대비 퇴보는 `reference_metrics`에만 기록되고
   `passed`에 영향 없음. ⑤추론 지연은 `warnings`에만 남고 차단 안 함 —
   `gate.yaml`의 `max_latency_s: null`(미실측이라 애초에 비워 둠).
4. **롤아웃 오차가 거의 0이면 자동 통과** — `check_rollout_recipe`: 초반 오차 평균이
   1e-6 미만이면 0으로 나누는 걸 피하려 ratio=1.0 자동 pass(ADR-0004에서 이어받음).
5. **모델별 결함 데이터가 없으면 그 물리 체크를 스킵** — `model_specific_violations
   =None`이면 실패 처리도 0 처리도 안 하고 "판단 안 함"으로 건너뜀(2026-11-11 사용자
   결정).
6. **재학습 루프 내부 CD는 가짜(깊이 값 복제)** — `retrain/execute.py::_eval_recipes`
   주석: "CD 미배선 — 깊이 값 복제(run_gate_negative.py와 동일 관례)". 반면
   `eval/metrics.py::cd_profile()`(W6-12/W8-01 벤치마크에 쓰인 것)은 진짜 기하 기반
   CD — **같은 이름 "CD"가 코드 경로에 따라 신뢰도가 다름.**

## B. 입력 검증의 2단계 예외

7. **물리적으로 말이 안 되는 값은 거부(422), 학습 범위 밖은 경고만** —
   `src/etchpipe/serve/app.py`의 `Recipe` 필드는 상식 밖 값만 거부.
   `src/etchpipe/serve/inference.py:142 is_in_range()`는 범위를 벗어나도 거부 안 하고
   `in_range=false`로만 표시(DOC-01).
8. **대시보드용 엔드포인트만 인증 예외** — `app.py:259,270,280`의 `/api/predict`,
   `/api/models/compare`, `/api/gate/history`는 `check_api_key`(`app.py:60`) 없이
   열려 있음(같은 출처 대시보드 편의 목적) — `/predict`·`/model`은 `x-api-key` 필요.

## C. 모델·시뮬레이션 알고리즘의 근사/예외

9. **t=0 마스크를 모든 스텝에서 재사용** — `train/rollout.py:230-235` — "식각은 안쪽
   으로만 진행(언더컷 제외)" 가정으로 깊이방향 마스크를 고정 — 보잉·언더컷 미측정
   한계(W8-02)의 실제 원인 코드.
10. **롤아웃이 완전히 발산해도 예외를 안 던지고 억지로 채점** — `train/rollout.py:
    223-228,254-264` — `outer_path`가 양 끝점을 못 찾으면 발산 좌표를 그대로 "곡선"
    으로 쓰고 `self_intersections=1`을 강제 기록 — 배치 전체가 죽는 대신 나쁘게라도
    채점.
11. **Chamfer/Dice 질의점만 Si 영역으로 제한, 비교 대상 곡선은 제한 안 함** —
    `eval/metrics.py:52-61 _restrict_to_silicon` — 마스크 영역 포함 시 surface-Dice가
    0.59→0.01로 왜곡되는 걸 막으려 질의점만 `y≤0`로 자름. 의도적 비대칭.
12. **SDF 부호 규약은 물리적 근거가 아니라 관례** — `etl.py:197` —
    "[제안: 부호 규약 — 레벨셋/그래픽스에서 흔한 '내부 음수' 관례를 따름, 다른 근거
    없음]" — 과장 방지용 자기 라벨링.
13. **materials.yaml이 과거 버그를 경고 주석으로 영구 보존** — `configs/etl/
    materials.yaml:7-8` — 초기 draft가 마스크/Si ID를 거꾸로 매핑했던 버그를 "다시
    반복하지 말라"는 주석으로 남김.

## D. 데이터·모델 버전관리의 예외

14. **골든셋은 held-out이 아니라 train의 부분집합** — `docs/data_card.md` §3.2 —
    일반화 평가가 아니라 고정 회귀 기준점이 목적임을 명시.
15. **FNO는 자동 게이트·레지스트리를 한 번도 거친 적 없음** — `grep -i fno
    registry/*.jsonl registry/manifest.json` 결과 0건 — 팀 평가 노트북으로만 "합격"
    확인(`runs/fno_tuned/metrics/`).
16. **레지스트리 이력은 승격된 시도만 기록, 실패한 시도는 고아로 남음** —
    `retrain/execute.py:320 if result["promoted"]:` 안에서만 `reg.register()` 호출 —
    BLOCK된 시도의 체크포인트(`runs/retrain/attemptN_*`)는 `history.jsonl`에 전혀
    안 남고 디스크에 고아로 남음(정리 스크립트 미구현, W8-07 향후 과제 #5 관련).
17. **외부 데이터 백업은 구버전으로 영구 고정** — GitHub Release `data-v1-n400`,
    Hugging Face `code5312/etch-ai-automation-data`가 W3-12 버그 수정 전 버전인 채로,
    재업로드 안 하기로 확정(W8-03 사용자 결정, `docs/data_card.md` §4).

## E. 저장소/빌드 설정의 예외

18. **`.gitignore` 블랭킷 규칙에 뚫어놓은 구멍들** — `.gitignore:83`
    (`!/runs/mgn_tuned/model.pt`), `:91-96`(`runs/fno_tuned`의 4개 파일만),
    `:22-23`(`golden/`·`tests/fixtures/`의 `.vtp`), `:17-18`
    (`reports/model_issues_{mgn,fno}.md`, `reports/*` 블랭킷의 예외).
19. **금지표현 검사는 `docs/` 전체를 빼지만 데이터카드·모델카드는 다시 넣음** —
    `scripts/check_forbidden_expressions.py:27 SCAN_FILES`, `:30 SELF_EXCLUDE`(검사
    스크립트와 그 테스트 파일은 자기 자신을 제외).
20. **ruff lint에서 두 곳을 의도적으로 제외** — `pyproject.toml:37-40
    extend-exclude`: `src/etchpipe/sim/sim_batch.py`(원본 보존),
    `runs/fno_tuned`(팀원 원본, "재구현하지 않는다" 원칙).
21. **ViennaPS(GPL-3.0)만 따로 라이선스 조건** — 저장소 코드는 Apache-2.0, ViennaPS
    포함 이미지에만 `NOTICE`로 별도 명시, 정적 링크 안 함(`README.md` License 절).

## F. 테스트의 예외

22. **`conftest.py`가 테스트 전용 API 키를 주입** — `tests/conftest.py:5` —
    `serve/app.py`가 import 시점에 `ETCH_API_KEY`를 요구해서, 테스트 수집 전에
    `os.environ.setdefault`로 가짜 키(`test-key-not-for-production`)를 심어둠(운영
    동작과의 유일한 차이).

## 요약

| 범주 | 개수 |
|---|---|
| A. 게이트 로직 | 6 |
| B. 입력 검증 | 2 |
| C. 모델·시뮬레이션 근사 | 5 |
| D. 데이터·버전관리 | 4 |
| E. 저장소·빌드 설정 | 4 |
| F. 테스트 | 1 |
| **합계** | **22**(중복 통합 전 조사 시점 24건 중 2건은 상위 항목에 통합) |

가장 중요하게 다시 봐야 할 3가지: **#6**(CD가 코드 경로마다 신뢰도가 다름 — 보고서
작성 시 어느 CD를 인용하는지 항상 명시해야 함), **#9**(언더컷 미측정의 직접 원인),
**#16**(레지스트리 이력 공백 — DB가 아니라 정리 스크립트로 해결 권장, 직전 대화
참고).
