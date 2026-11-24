# W9-01: 재학습 산출물 생애주기 정리 스크립트

## 1. 문제

`retrain/execute.py::run()`은 승격 시 `model.pt`를 `Path.replace()`(이동)로
`runs/retrain/<model_id>/`로 옮긴다. 즉 원본 `runs/retrain/attempt<N>_<ts>/` 디렉터리는
승격 여부와 무관하게 루프가 끝나면 항상 소비된 찌꺼기다 — 재학습 루프는 매 시도 원본
`champion_ckpt`에서 다시 시작하므로(`retrain/loop.py::run_retrain_loop`) 과거 attempt의
체크포인트를 다시 읽는 코드 경로가 없다. 실패한 시도는 `history.jsonl`에 기록만 되고
디스크 정리는 전혀 안 된다(`register()`가 `if gate_result.passed` 안에서만 카드를 씀).

`data/drift/processed/<hash>/`도 비슷하다 — `DriftState.expand_train()`이 train을
40→100으로 늘릴 때마다 새 `dataset_hash` 디렉터리가 생기는데, 이전 해시는 레지스트리의
현재/과거 어떤 모델도 참조하지 않게 되면 다시 안 쓰인다.

## 2. 구현

`scripts/cleanup_retrain_artifacts.py` — 순수 함수 3개 + CLI:

- `find_attempt_dirs(retrain_root, min_age_seconds, now)`: `attempt`로 시작하는
  디렉터리만(레지스트리 모델 디렉터리는 `model_id`로 이름 붙어 안 걸림), 최근
  `min_age_seconds`(기본 30분) 이내 수정된 건 제외 — 진행 중인 루프 보호.
- `referenced_dataset_hashes(registry_root)`: `registry/*/card.json`의 `dataset_hash`
  전부(현재 챔피언뿐 아니라 과거 모델 포함) 수집.
- `find_orphan_drift_dirs(drift_processed_root, keep_hashes)`: 위 집합에 없는 해시
  디렉터리만.

기본은 dry-run(목록 출력만), `--apply`로만 실제 `shutil.rmtree`. 둘 다 `/data/`,
`/runs/`, `/registry/`(저장소 루트 `.gitignore`)로 git 추적 대상이 아니라 되돌릴 git
이력도 없음 — 그래서 dry-run 기본값과 나이 가드를 넣었다[제안].

## 3. 테스트

`tests/unit/test_cleanup_retrain_artifacts.py` — 7개, `tmp_path`로 격리, `etchpipe`
패키지(torch/physicsnemo 필요)에 의존하지 않고 `scripts/`를 `sys.path`에 직접 추가해서
import. 로컬 `python -m pytest`로 실행 가능(Docker 불필요) — 전부 PASS.

```
$ python -m pytest tests/unit/test_cleanup_retrain_artifacts.py -v
7 passed in 0.24s
$ python -m ruff check scripts/cleanup_retrain_artifacts.py tests/unit/test_cleanup_retrain_artifacts.py
All checks passed!
```

## 4. 실제 저장소 대상 dry-run 결과 [확인, 2026-11-23]

```
$ python scripts/cleanup_retrain_artifacts.py --min-age-minutes 0
[runs/retrain/attempt*] 1건 (0.0분보다 오래됨)
  - runs\retrain\attempt1_1794510586
[data/drift/processed/<고아 해시>] 0건 (보존: ['26a29dcca7da', '4b8c0ce1bc98'])

(dry-run - 실제 삭제는 --apply)
```

- `attempt1_1794510586`은 실제 2026-11-12 승격 사건의 원본 디렉터리(`model.pt`는 이미
  `runs/retrain/mgn_retrain_20261112T212130/`로 이동됨, `metrics.json`만 남은 빈 껍데기
  — `ls` 확인). 삭제해도 정보 손실 없음(승격 이력은 `reports/evidence/
  retrain_20261112T212130.json`과 `registry/mgn_retrain_20261112T212130/card.json`에
  이미 남아 있음, `metrics.json`의 에폭별 loss만 없어짐).
- 드리프트 처리 디렉터리는 아직 1개뿐(`26a29dcca7da`)이라 고아가 0건 — 실제 운영에서
  `expand_train`이 한 번도 안 불렸기 때문(W6-06: attempt 1에서 바로 PASS).
- **`--apply` 실행 완료(2026-11-24, 사용자 확인 후)**: `runs/retrain/attempt1_1794510586`
  삭제. 삭제 전 `metrics.json`(에폭별 loss 3줄, 다른 어떤 파일에도 중복 저장 안 돼
  있었음 — 코드 전체에 이 파일을 다시 읽는 곳이 없어 기능상 영향 없음, 다만 raw
  진단 데이터 자체는 사라짐[정정: 최초 작성 시 "정보 손실 없음"이라 쓴 건 부정확했음])
  확인 후 진행. 삭제 후 `runs/retrain/`엔 `mgn_retrain_20261112T212130/`(승격된
  챔피언)만 남음, `registry/manifest.json`의 `current`·`models` 변화 없음(영향 없음
  확인).

## 5. 전체 테스트 스위트 상태

`tests/unit/` 전체는 `etchpipe` 패키지(torch/physicsnemo) 임포트가 필요해 로컬
Python(패키지 미설치)에서는 21개 파일이 collection error — `etch-train:0.1` Docker
이미지가 필요한 기존 환경 제약(CLAUDE.md "환경 3분리")이고, 이번 변경과 무관함[확인,
`test_registry.py` 단독 실행 시 `ModuleNotFoundError: No module named 'etchpipe'`로
재현, Docker 데몬 현재 미기동이라 `etch-train:0.1` 재확인은 못 함[미확인]].
