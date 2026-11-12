# W5-02: 평가 진입점 `eval.py` ([CPU/매핑])

- 날짜: 2026-11-10
- 코드: `src/etchpipe/eval/eval.py`(신규)
- 테스트: `tests/unit/test_eval.py`(신규 8건)

## 구현

`run_eval(model, graphs_root, raw_dir, val_ids, norm, out_dir, steps=(5,10,20), n_plots=5)`:
val 전체(정렬된 순서 — 결정적)에 대해 `rollout.rollout_val_error_by_step`(W5-01)로 스텝
5·10·20과 각 run의 최종 스텝(`actual_k`, run마다 다름)을 계산해 `metrics.json`에
`by_step`(스텝별 집계, 그 스텝까지 못 가는 짧은 run은 자동 제외)·`final_step`(전체 집계)·
`per_run`(run별 전체 스텝)으로 남긴다. 앞 `n_plots`개 run은 최종 스텝 예측 vs 정답
오버레이를 `plots/<run_id>.png`로 저장(matplotlib, `Agg` 백엔드 — 디스플레이 없는
컨테이너/CI에서도 저장만 되면 됨).

`load_model_a`: `scripts/run_rollout_comparison.py`의 체크포인트 로드 로직(processor_size
역산, `net.` 접두사 지원, W4-06/W4-04에서 발견한 버그 수정들)을 재사용 가능한 함수로 옮김.
`MeshGraphNetModel`은 함수 안에서 지연 임포트(physicsnemo 없는 호스트에서도 나머지는
토이 모델로 테스트 가능 — `train_main.build_model`과 같은 이유).

`main()`: config → `dataset_hash` → `data/processed/<hash>/{graphs,stats.json}`,
`data/splits.json["val"]` 로드 → `run_eval` → `metrics.json`에 `checkpoint`/
`dataset_hash`/`seed`/`git_sha`/`packages`/`hardware`(플랫폼·torch 버전·device·cuda 여부)
추가. CLI: `python -m etchpipe.eval.eval <checkpoint> [--config] [--out-dir] [--device]
[--steps] [--n-plots] [--seed]`.

### 버그 발견·수정: `git_sha()` 실패가 이미 계산된 지표를 날림

처음 실제 체크포인트로 끝까지 돌려보니(아래 "실행" 참고), val 40개 롤아웃을 **전부
계산한 뒤** `metrics.json`을 쓰기 직전 `git_sha()`가 git 미설치 환경(`etch-train:0.1`
컨테이너, `test_run_train.py`의 기존 3건과 같은 한계)에서 예외를 던져 **아무것도
저장되지 않고 전체가 날아갔다**. `run_train.py`의 `init_run()`은 같은 문제가 있어도
학습 "시작 전"이라 다시 돌리면 되지만, 평가는 비용이 다 든 "끝"에서 터지는 차이가 있다
— `safe_git_sha()`(try/except, 실패 시 `"[미확인: git 명령 실패]"`)로 감싸서 메타데이터
하나 때문에 실제 지표를 버리지 않게 고침.

## 테스트

토이 모델(`nn.Linear`) + 실제 fixture(`r0000~r0002`, `actual_k=3`)로 집계·플롯·구조·
`git_sha` 대체 로직 검증:
- `test_run_eval_writes_metrics_structure_and_plots`: `by_step`/`final_step`/`per_run`
  구조, 앞 2개 run만 플롯 생성 확인.
- `test_run_eval_requested_step_beyond_actual_k_is_skipped`: 요청 스텝(10)이 `actual_k`(3)
  보다 크면 `by_step`에 안 잡히고 최종 스텝(3)만 남는지 확인.
- `test_run_eval_is_deterministic_across_repeated_calls`: 같은 시드로 두 번 돌려 결과가
  완전히 같은 JSON인지 확인.
- `test_safe_git_sha_falls_back_*`(3건): `FileNotFoundError`/`CalledProcessError` 둘 다
  대체 문자열로 떨어지는지, 정상일 때는 실제 값을 그대로 돌리는지.

```
docker run --rm --memory=2g -v "<repo>:/app" -w /app etch-train:0.1 bash -lc "
python -m ruff check src/ tests/   # All checks passed
PYTHONPATH=src python -m pytest tests/unit -q --ignore=tests/unit/test_serve.py
"
→ 121 passed, 3 failed(기존 git 미설치 환경 한계, 오늘 변경과 무관)
```

## 실행: 실제 체크포인트(epoch=99)로 val 40개 끝까지

```
PYTHONPATH=src python -m etchpipe.eval.eval \
  runs/20261110T021927908408_b369f6/ckpt_last.pt \
  --out-dir reports/eval/W5-02_sanity_epoch99 --n-plots 3
```

```
n_val=40
final_step: chamfer mean=2.182 median=2.383 / depth_error mean=3.007 median=2.521
```

`final_step` 수치가 `reports/evidence/W4-06_rollout_vs_baseline.md`의 기존 epoch=99
결과(Chamfer 중앙값 2.383, 깊이오차 중앙값 2.521)와 정확히 일치 — 새 진입점이 기존
스크립트와 같은 계산을 재현한다는 교차 검증.

**스텝별(처음으로 실측, W5-01 "판단 기준" 3번이 비어 있던 부분 채움) [확인]**:

| 스텝 | Chamfer 중앙값 | 포함된 run 수(그만큼 긴 run만) |
|---|---|---|
| 5 | 0.406 | 40/40 |
| 10 | 0.892 | 34/40 |
| 20 | 1.989 | 21/40 |
| 최종(평균 actual_k) | 2.383 | 40/40 |

오차가 스텝이 늘수록 거의 선형으로 커지는 패턴(급격한 스파이크는 아님)으로 보임
[제안, 급증 여부는 중간 스텝이 3개뿐이라 단정하기엔 이름 — 전체 스텝별 시계열을 보면
더 정확히 판단 가능]. `reports/model_issues_mgn.md`에 반영.

`git_sha`: 이 실행도 git 미설치 컨테이너에서 돌려서 `"[미확인: git 명령 실패]"`로
기록됨 — 지표 자체는 정상 저장(위 버그 수정이 바로 이 상황).

## 다음 (미착수)

- `plots/`의 PNG 자체를 사람이 눈으로 확인하지는 않음(파일 생성 여부만 테스트) —
  필요하면 직접 열어서 오버레이가 그럴듯한지 봐야 함 [미확인].
- matplotlib이 `pyproject.toml`의 `dependencies`에 명시돼 있지 않음(지금은 `etch-train:0.1`
  컨테이너에 다른 패키지가 끌고 들어온 버전 3.11.2가 이미 있어서 그냥 썼음) — 깨끗한
  환경 재현성(정의서 §4 목표⑤)을 위해 명시적으로 추가할지 사용자 확인 필요
  [미확인, CLAUDE.md 금지 "사용자 확인 없이 의존성 변경"].
- FNO(모델 B)는 아직 없어 이 진입점은 모델 A만 지원.
