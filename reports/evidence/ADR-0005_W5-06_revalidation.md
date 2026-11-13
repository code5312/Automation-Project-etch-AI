# W5-06 재검증: ADR-0005 신규 경로(레시피별 통과율 + 윌콕슨) 부정 테스트

ADR-0005 2단계(`retrain/execute.py::make_gate_fn`이 더 이상 `run_gate`가 아니라
`check_recipe_pass_rate`/`check_no_regression`을 직접 쓰도록 바뀜)로 인해, W5-06
부정 테스트 6종이 기존 `run_gate`만 검증하고 실제 드리프트 승격 경로가 쓰는 새 함수는
한 번도 검증 안 된 상태였다. `scripts/run_gate_negative.py`를 확장해 같은 6종 시나리오를
**기존 경로와 신규 경로 양쪽으로 똑같이** 재실행했다.

## 변경
- `scripts/run_gate_negative.py`: `evaluate_model`이 IoU·HD95도 같이 계산(`check_shape=True`).
  `_new_path_gate()`(신규 — 실제 `make_gate_fn`과 같은 순서로 관문 A → 물리 → 관문 C 적용,
  관문 C③(val 40)은 CI에 `data/raw`가 없어 건너뜀) 추가. 기존 4개 모델 케이스를
  `[기존 run_gate]`/`[ADR-0005 신규 경로]` 쌍으로 확장(해시 위조 2종은 `run_gate` 전용
  기능이라 그대로 유지, 쌍으로 안 만듦).

## 실행 (`etch-train:0.1`, 실제 champion 체크포인트 + golden/raw 10개)
```
docker run --rm -v $(pwd):/app -w /app etch-train:0.1 python scripts/run_gate_negative.py
-> gate-negative 통과: 10개 케이스 모두 기대대로 동작
```
두 번 반복 실행해 수치 안정성 확인(치명적 변동 없음).

| 케이스 | 기존 run_gate | ADR-0005 신규 경로 |
|---|---|---|
| 정상 모델(mgn_tuned) | PASS | **BLOCK**(80%<90%, 기대값 — 아래 참고) |
| 노이즈 모델 | BLOCK | BLOCK(0%) |
| 1에폭(저학습) 모델 | BLOCK | BLOCK(30%) |
| 시뮬레이터 버전 변경 모델 | BLOCK | BLOCK(0%) |
| dataset_hash 불일치 | exit_code=2 | (해당 없음, run_gate 전용) |
| 골든셋 변조 | exit_code=2 | (해당 없음, run_gate 전용) |

## 핵심 발견 — 버그 아님, 실측으로 확인한 ADR-0005의 채택 근거

**정상 모델(mgn_tuned)이 신규 경로에서 BLOCK**됐다(golden 10개 중 8개만 레시피별
3조건 통과, 80%<90%). 원인 분석:
- golden 10개는 설계상 "중심5·경계3·극단2"로 **일부러 어려운 레시피를 섞은 소표본**
  (`docs/project-design.md` C5, `docs/adr/0004-...md` 참고).
- 기존 `run_gate`(평균 집계)는 10개 전체를 모아 평균 오차 하나로 판정 — 2개의 실패가
  나머지 8개 평균에 묻혀 PASS가 나온다.
- ADR-0005 신규 경로(레시피별 판정 + 통과율)는 평균에 안 숨고 그대로 드러낸다.

이건 정확히 ADR-0005가 채택된 이유("평균 집계가 개별 레시피 실패를 가린다")를 실측으로
재확인한 것 — 버그가 아니라 설계대로 동작한 결과다[사용자 결정 2026-11-14, "80%로
예상대로"로 문서화하고 넘어가기로 함].

**이 결과가 "지금 챔피언이 실제로는 불합격"이라는 뜻은 아니다**: 실제 드리프트 승격
판정은 이 golden-10이 아니라 `golden_drift10+val10`(20개) 또는 `val 40`을 쓴다. 90%
기준 자체가 val 40 기준 Clopper-Pearson 신뢰하한으로 도출된 값이라(문서 §5), N=10·
편향 표본에 그대로 적용하면 이렇게 불안정하게 나올 수 있다는 한계로 기록해 둔다.

## 미확인 / 남은 작업
- 실제 운영 규모(golden_drift10+val10=20개, val 40개)로는 이 "정상 모델 80% BLOCK"
  현상이 재현되는지 확인 안 함 — 로컬/별도 실행 환경에서 돌려야 함(데이터는
  `data/drift/`·`data/raw`에 이미 있음, CI에는 없음).
- 관문 C③(val 40 재확인)은 이 재검증에서 완전히 생략됨(`val_pass_rate_result=None`) —
  CI 데이터 제약 때문. 실제 재검증은 2단계 작업의 execute.py 엔드투엔드 스모크(대화
  이전 세션)에서만 확인됨.
- CI(GitHub Actions `gate-negative` job)에서 재확인 필요 — 로컬 Python/패키지 버전이
  CI와 100% 동일하지 않음. 실행 시간이 늘어난 것(새 경로 때문에 IoU/HD95 계산 추가)도
  CI 타임아웃에 영향 있는지 확인 필요.
