# W5-05/W5-07: 게이트 완성 + 레지스트리 완성 ([CPU/매핑])

- 날짜: 2026-11-11
- 코드: `src/etchpipe/gate/gate.py`, `src/etchpipe/registry/registry.py`
- 테스트: `tests/unit/test_gate.py`(신규 12건), `tests/unit/test_registry.py`(신규 11건)

## 배경

W2-07에서 게이트는 규칙 ①(골든 Chamfer)·④(롤아웃 후반 급증)만, 레지스트리는 승격·
champion 포인터·`history.jsonl`만 있었다. "첫 정상 모델"이 아직 없어(팀원이 Hugging
Face에서 MGN/FNO 학습 중, 모델 품질과 무관한 부분부터 먼저 진행하기로 함) 체크포인트
의존 없는 두 항목(W5-05, W5-07)을 먼저 완성했다.

## W5-05: 게이트 규칙 ②③⑤ + 챔피언 없음 케이스 + `gate_result.json`

설계서 §3 C5 규칙(초기, [제안, 베이스라인 후 보정]):

- **②** `check_cd_depth_error(depth_error_pct, cd_error_pct, max_pct=5.0)`: 두 리스트의
  평균이 `max_pct` 이하인지 [제안, 평균 기준 — 설계서는 "≤5%"만 명시]. 퍼센트 오차 자체는
  `eval.py`가 아직 안 내서(지금은 절대 오차 µm만) 호출부가 계산해서 넘겨준다는 전제.
- **③** `check_champion_regression(current_mean_error, champion_mean_error, max_regression=0.10)`:
  챔피언이 `None`이면(체크리스트 "챔피언 없음 케이스") 비교 대상이 없어 자동 통과.
- **⑤** `check_latency(latency_s, max_latency_s)`: 상한 초과 시 **경고 문구만** 반환, 차단
  안 함 — `GateResult.warnings`(신규 필드)에만 쌓이고 `passed`에 영향 없음(설계서·체크리스트
  "경고"라는 표현 그대로).
- `run_gate`: 새 인자들을 `None`으로 두면(안 넘기면) 해당 검사를 건너뛰어 **기존 호출부
  (`scripts/run_gate_negative.py`, 기존 테스트)는 그대로 동작**(하위 호환).
- `write_gate_result(result, out_path, thresholds)`: 설계서 §3 C5 출력(`gate.json`)/
  체크리스트 "`gate_result.json`" — 판정+근거 수치(`reasons`/`warnings`)+호출부가 실제
  쓴 임계치를 JSON으로 남긴다.

## W5-07: 레지스트리 — 카드·valid_range·manifest·무결성·롤백

설계서 §3 C6:

- `register(model_id, artifact_path, gate_result, *, dataset_hash, config, git_sha,
  metrics, valid_range)`: **PASS일 때만** `registry/<model_id>/{card.json,
  valid_range.json}`을 쓰고 승격(`promote` 재사용) — 설계서 §10 "PASS 판정이 아닌
  모델은 `registry/`에 쓰이지 않는다"를 그대로 반영. BLOCK은 `history.jsonl`에만 남는다
  (이건 W2-07부터 있던 감사 로그라 §10의 "쓰기 제한" 대상이 아님 — 안 그러면 기존
  `test_promote_on_fail_does_not_set_champion`과 모순됨, 코드 작성 전에 설계서 재확인으로
  발견).
- `sha256_file`/`verify_artifact`: 설계서 §3 C6 "무결성" — 카드에 SHA-256을 남기고 대조.
- `compute_valid_range(raw_dir, run_ids)`: 실제 train split의 Θ 실측 min/max [제안,
  정의서 §7.1 표의 샘플링 상한이 아니라 "실제로 학습에 쓰인 범위"].
- `rollback(model_id)`: 과거 `register`로 등록된(= `card.json` 있는) model_id로 champion을
  되돌림 — 게이트 재실행 없는 수동 복구라 `GateResult` 불필요. 등록 안 된 id는
  `ValueError`.
- `_write_manifest()`: `register`/`promote`/`rollback` 호출마다 `manifest.json`(현재
  champion + 카드 있는 model_id 목록)을 자동 갱신.

## 테스트

- 게이트: ②③⑤ 각각 통과/차단(또는 경고) 경계값, 챔피언 없음 자동통과, `run_gate`
  통합(각 규칙 단독 차단 확인), `write_gate_result`가 임계치 포함해 JSON으로 남는지.
- 레지스트리: `sha256_file`이 `hashlib` 기준값과 일치, `compute_valid_range`가 실제
  fixture(`r0000~r0002`) 3개 run의 `ion_flux` min/max를 정확히 계산(3.455.../11.386...),
  PASS 시 카드+valid_range.json 생성·승격, **BLOCK 시 디렉터리 자체가 안 생김**(설계서
  §10 직접 검증), `verify_artifact`가 파일 변조를 감지, `rollback` 성공/미등록 id 거부.

```
docker run --rm --memory=2g -v "<repo>:/app" -w /app etch-train:0.1 bash -lc "
python -m ruff check src/ tests/   # All checks passed
PYTHONPATH=src python -m pytest tests/unit -q --ignore=tests/unit/test_serve.py
"
→ 140 passed, 3 failed(기존 git 미설치 환경 한계, 오늘 변경과 무관), 신규 23건 포함
```

## 다음 (미착수)

- W5-04(게이트 임계값 산출)가 "첫 정상 모델"을 기다리는 중 — 팀원의 Hugging Face
  학습 결과나 우리 쪽 `ckpt_best.pt` 재학습 중 하나가 나와야 `max_chamfer`/
  `max_cd_depth_pct` 등 실제 숫자를 `configs/pipeline/gate.yaml`에 고정할 수 있음.
- `eval.py`가 아직 %오차(depth_error_pct/cd_error_pct)를 안 냄 — 지금은 절대오차(µm)만
  계산하므로, W5-04/06에서 실제로 게이트를 돌리려면 `eval.py`에 % 변환을 추가하거나
  호출부에서 직접 계산해야 함 [미확인, 다음 연결 지점].
- W5-06(부정 테스트 6종)은 이 두 항목이 끝나 게이트 로직이 완성됐으니 다음 순서로
  진행 가능 — 다만 "1에폭 모델"·"시뮬레이터 버전 변경 데이터로 학습한 모델" 두 경우는
  실제 모델이 있어야 의미가 있어 일부는 여전히 체크포인트 대기.
