# ADR-0010: 드리프트 감지→재학습 자동 트리거 + champion 동적 조회

- 날짜: 2026-11-25
- 상태: **승인(사용자 결정 2026-11-25, W9-06 진행 지시 — GitHub Actions 캐시 방식
  명시적 선택, 두 차례 재확인 후 동일 답변)**
- 영향 파일: `src/etchpipe/registry/registry.py`(신규 `resolve_artifact_path`),
  `src/etchpipe/serve/inference.py`(기존 로직을 공용 함수로 교체, 동작 불변),
  `src/etchpipe/retrain/execute.py`(신규 `current_champion_ckpt`, `run()` 수정),
  `scripts/run_drift_monitor.py`(champion 동적 조회로 교체),
  `.github/workflows/drift-retrain-schedule.yml`(신규),
  `docs/project-checklist.md` W9-06.
- CLAUDE.md 작업규칙 6: `run()`의 기본 champion 소스가 바뀌는 행동 변경(새 선택 인자가
  아니라 기존 동작 자체가 바뀜)이라 ADR 선행.

## 배경 — 두 개의 독립적인 문제

W9-06("드리프트 감지→재학습 자동 트리거") 설계 중 그 전제 자체에 있던 두 가지 공백을
발견했다. 둘 다 고치지 않으면 자동 트리거를 쌓아도 의미가 없다고 판단해 이 ADR
하나로 같이 묶는다.

### 문제 1: champion이 고정 baseline에 박혀 있었다

`retrain/execute.py::CHAMPION_CKPT`와 `scripts/run_drift_monitor.py::MGN_TUNED_CKPT`가
둘 다 `runs/mgn_tuned/model.pt`(최초 수동 등록 baseline, `mgn_tuned_ep2`)로
하드코딩돼 있었다. 레지스트리가 이미 다른 모델(예: 실존하는
`mgn_retrain_20261112T212130`)을 champion으로 가리키고 있어도 이 둘은 그걸 몰랐다.
서빙 쪽(`serve/inference.py::load_champion`)은 이미 W7-01(2026-11-13)에서 같은 문제를
겪고 고쳤었다("champion이 재학습 산출물로 바뀌며 `/predict`가 503을 내던 버그") — 그
수정이 재학습 루프·모니터로는 전파되지 않았던 것.

**영향**: 자동 재학습이 한 번이라도 승격되면, 그 뒤로 `python -m etchpipe.retrain run`과
드리프트 모니터는 영원히 그 이전(구버전) baseline만 보고 반복한다 — 승격 자체가
무의미해진다. 자동 트리거를 추가하면 이 괴리가 사람 눈에 안 보이게 매주 조용히
반복된다.

### 문제 2: GitHub 호스팅 러너는 영속 디스크가 없다

`data/`·`runs/`·`registry/`는 전부 `.gitignore` 대상이다(`runs/mgn_tuned/model.pt`
등 몇 개 예외 제외). 기존 `retrain-demo.yml`은 "registry에 쓰지 않는다"로 설계해 이
문제를 원천적으로 피했다 — 진짜 자동 트리거(`retrain.execute.run()`을 실제로
호출해 실제 registry를 갱신)는 이 회피를 쓸 수 없다. GitHub 호스팅 러너
(`ubuntu-latest`)는 매 실행마다 완전히 새 디스크에서 시작해, 스케줄 워크플로우가
돌 때마다 지금까지 쌓인 champion/registry 상태가 없는 빈 디스크를 보게 된다.

## 결정

### 결정 1 — champion 동적 조회로 교체

`registry/registry.py`에 공용 함수 `resolve_artifact_path(model_id, legacy_paths,
retrain_runs_root)` 추가 — `serve/inference.py::_artifact_path_for`가 이미 하던 로직
(최초 등록분은 고정 매핑, 그 외는 `<retrain_runs_root>/<model_id>/model.pt` 규칙)을
그대로 추출해 공용화. `serve/inference.py`는 이 함수로 위임하도록 교체(기존
monkeypatch 테스트 7건 전부 그대로 통과 확인 — 동작 불변).

`retrain/execute.py`에 `current_champion_ckpt()` 추가 — `Registry(REPO_ROOT /
"registry").current_champion()`로 model_id를 얻고 `resolve_artifact_path`로 실제
경로를 찾는다. 레지스트리에 champion이 전혀 없으면(최초 부트스트랩 전)
`CHAMPION_CKPT`(baseline, git에 커밋된 파일이라 항상 존재)로 후퇴. `run()`이 이제
`CHAMPION_CKPT` 상수 대신 이 함수를 호출 — fine-tune 시작점과 게이트 비교 기준
(`make_gate_fn(state, champion_ckpt=champion_ckpt)`) 둘 다 실제 현재 champion을
쓴다. `scripts/run_drift_monitor.py`도 동일하게 교체, 어떤 champion을 체크했는지
결과 JSON에 `champion_ckpt` 필드로 기록(추적 가능하게).

**범위 제외**: `scripts/run_retrain_demo.py`/`run_gate_negative.py`/
`run_model_comparison.py`는 바꾸지 않는다 — 데모·회귀 비교·벤치마크는 "고정된 알려진
기준점"이 필요한 성격이라(실제 운영 champion과 무관하게 재현 가능해야 함), 동적
조회로 바꾸면 오히려 재현성을 해친다.

### 결정 2 — GitHub Actions 캐시로 상태 지속(사용자 선택, 명시적 한계 인정)

사용자에게 세 가지 선택지(①GH Actions 캐시 ②self-hosted 러너 전제 ③자동 실행 없이
감지·알림만)를 제시했고, **①GH Actions 캐시**를 선택(2026-11-25, 재확인 질문에도
동일 답변). 설계:

- `.github/workflows/drift-retrain-schedule.yml`(신규): `schedule`(매주 월요일 03:00
  UTC[제안, 임의 기본값]) + `workflow_dispatch`(수동 테스트용).
- 드리프트 모니터링 raw 시뮬레이션(ViennaPS, 레시피당 8~47분 — 10개면 수시간
  [확인, `run_drift_sim.py` 독스트링])은 `configs/sim/live.yaml` 해시로 캐시
  (`actions/cache@v4`) — 설정이 안 바뀌면 재시뮬레이션 생략.
- `registry/`·`runs/retrain/`는 `actions/cache/restore`(접두사 `restore-keys`로 가장
  최근 저장분 복원) + `actions/cache/save`(매 실행마다 새 키로 저장, `if: always()`)
  조합으로 "누적 스냅샷"처럼 이어간다.
- `check_drift` 결과(`reports/eval/W6-02_drift_monitor.json`)의 `triggered` 필드를
  `jq`로 읽어 `true`일 때만 `python -m etchpipe.retrain run` 실행.
- 결과(드리프트 모니터 JSON + 재학습 증거 JSON)는 워크플로우 아티팩트로 업로드.

**2026-11-25 보완(실행 전 코드 리뷰로 발견, 사용자 요청으로 수정)**: 최초 설계는
`data/drift_monitor/`(모니터링용 10개)만 캐시했는데, `retrain.execute.run()`의
`DriftState()`가 요구하는 **재학습용** 드리프트 데이터(`data/drift/`, train 40/
val 10/golden_drift 10 = 60개, 모니터링용과 완전히 별개 레시피셋)와, 게이트 C
(val-40 재확인)가 요구하는 **원본** 400개 데이터셋(`data/raw`+
`data/processed/4b8c0ce1bc98`+`data/splits.json`)을 빠뜨려, `triggered=true`가
뜨는 순간(또는 게이트 A를 통과하는 순간) `FileNotFoundError`로 죽는 구조였다 —
champion 동적 조회(결정 1) 코드에 도달하기도 전에 실패. 보완:

- `data/drift`: `data/drift_monitor`와 같은 방식(`configs/sim/live.yaml` 해시 키
  캐시)으로 추가, 캐시 미스 시 `scripts/run_drift_data_gen.py`로 자동 재생성(2~6시간
  — `triggered`일 때만 돌게 조건을 걸어 평소엔 불필요한 재시뮬레이션을 피함).
- `data/raw`+`data/processed`+`data/splits.json`: **2026-11-25 2차 보완(사용자
  지시, 방법 1 채택)** — 캐시 미스 시 ViennaPS 재시뮬레이션(400개, 불가능) 대신
  이미 공개된 GitHub Release(`data-v1-n400` 태그, `etch-raw-n400.zip`, 42MB)를
  `gh release download`로 받아 자동 복구한다[확인, 2026-11-25 — 자산을 실제로
  받아 구조 확인: `<run_id>/meta.json` 400개(ood0000~19, 나머지 r####) +
  `recipes.json`]. 받은 뒤 `scripts/gen_splits.py`(raw 불필요, 고정 seed 순수
  계산 — [확인, 코드 직접 읽음])로 `data/splits.json` 생성, W3-04 당시와 동일한
  명령(`process_dataset(..., max_workers=4)`)으로 ETL. 결과 `dataset_hash`가
  `execute.py::VAL40_GRAPHS_ROOT`가 하드코딩한 `4b8c0ce1bc98`와 다르면
  (`data/processed/4b8c0ce1bc98` 디렉터리 부재) 워크플로우가 `::error::`로 명시적
  실패 — 조용히 틀린 채 넘어가지 않게.
  이 자산은 W3-12 버그 수정 전 버전(ADR-0007)이지만, 그 버그는 OOD 레시피만
  건드렸고 게이트 C는 `val` 분할만 쓴다(`_val40_run_ids`, OOD 전혀 안 씀) — 이
  용도엔 무관[확인, W3-12 범위 재확인].

**명시적으로 인정하는 한계**[제안, 전부 실제 스케줄 실행으로 재확인 전]:

1. **진짜 영속 저장소가 아니다** — GitHub의 캐시 LRU 정책(저장소당 10GB, 7일 미접근
   시 삭제)에 그대로 종속된다. 캐시가 지워지면 이 워크플로우는 아무 경고 없이
   "최초 실행"처럼 재부트스트랩한다(champion이 baseline으로 후퇴).
2. **CI의 registry와 사용자 로컬 PC의 registry가 분리된, 서로 다른 레지스트리가
   될 수 있다** — 이 워크플로우가 자동으로 승격한 모델은 사용자의 로컬 디스크에는
   나타나지 않는다(반대도 마찬가지). 통합(단일 진짜 champion)은 이 ADR의 범위가
   아니다 — 그러려면 실제 영속 스토리지(self-hosted 러너 또는 외부 저장소)가
   필요하고, 사용자가 명시적으로 이 비용을 피하는 쪽을 선택했다.
3. **콜드 캐시(`live.yaml` 변경 직후) 시뮬레이션이 GitHub Actions 기본 작업 시간
   한도(6시간)를 넘길 수 있다** — 모니터링 10개(최대 7.8시간)뿐 아니라 보완된
   재학습용 60개(`data/drift`, 최대 수배)도 같은 위험을 가진다. 이 경우 워크플로우가
   중간에 강제 종료되고(`if: always()` 저장 단계가 못 돌 수 있음) 데이터가 일부만
   남을 수 있다 — 재시도(다음 스케줄 또는 수동 `workflow_dispatch`)가 필요하다.
   이번 ADR에서 해결하지 않음(시간 상자를 쪼개 여러 실행에 나누는 설계는 범위 밖,
   필요해지면 별도 ADR).
4. **`data/raw`(원본 400개)는 2026-11-25 2차 보완으로 GitHub Release 다운로드
   경로가 생겼지만, 그 경로 자체에도 한계가 있다**[제안, 실제 CI 실행으로 재확인
   전]: (a) Release 자산이 삭제·태그 변경되면 다운로드가 실패한다(현재는 사용자
   팀 소유 공개 Release라 통제 가능하지만 외부 요인). (b) ETL 재현이 정확히
   `4b8c0ce1bc98`를 만든다는 보장은 "같은 입력(raw+splits)이면 같은 해시"라는
   `process_dataset`의 결정론 가정에 의존한다 — `THETA_KEYS`나 ETL 로직이 그동안
   바뀌었으면(이력상 바뀐 적 없음, 2026-11-25 기준) 다른 해시가 나오고, 이 경우
   워크플로우가 `::error::`로 명시적으로 실패한다(조용히 틀리지 않음, 위 결정 2
   참고) — 그 실패를 해결하려면 `VAL40_GRAPHS_ROOT` 쪽을 사람이 다시 봐야 한다.

## 비용·트레이드오프

- 결정 1(champion 동적 조회)은 위험이 작다 — 기존 `serve/inference.py` 패턴을 그대로
  재사용, 단위테스트로 세 경로(baseline/재학습 산출물/찾을 수 없음) 모두 확인.
- 결정 2(스케줄+캐시)는 "자동화는 됐지만 신뢰도는 낮다" — 위 한계 3가지를 사용자가
  이미 알고 선택했다. 실제 스케줄 실행으로 캐시 적중률·콜드런 소요 시간을 재확인한
  적은 없음[미확인] — 다음 실제 스케줄 실행(또는 `workflow_dispatch` 수동 테스트) 때
  재확인 필요.

## 대안 (기각)

- **self-hosted 러너 전제**: 사용자가 유지비(전력 설정 켜둔 PC)를 감당하지 않는 쪽을
  선택(두 번째 재확인에서도 동일 선택) — 기각.
- **자동 실행 없이 감지·알림만(GitHub Issue 등)**: 체크리스트 원문이 요구한
  "`retrain.execute.run()` 호출에 연결"에 못 미침 — 기각(사용자가 캐시 방식으로
  실제 자동 실행을 선택).
- **champion 조회를 손대지 않고 트리거만 추가**: 문제 1을 그대로 둔 채 자동화를
  쌓으면 자동화가 매주 조용히 틀린 걸 반복하게 됨 — 기각, 이번 ADR에 같이 포함.
