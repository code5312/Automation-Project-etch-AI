# W8-03: README(설치→데모 한 줄 실행→증거 ①② 재현), 데이터 카드·모델 카드 최종

## 1. README

- 추가: **Demo(한 줄 실행)** 섹션 — 기존 README가 스스로 "단일 명령으로 정리돼 있지
  않다"고 적어 둔 공백을 메움. `docker run --rm ... etch-serve:0.1 python
  scripts/run_retrain_demo.py` 한 줄로 실제 champion+실제 ViennaPS+실제 게이트를
  도는 C8 시연을 실행 — 이미 W7-07/이전 주차에서 실제로 완주까지 검증된 명령이라
  새 실행 없이 그대로 인용. 예상 소요시간(1차 PASS면 수분, 4회 전부면 수시간)과
  `reports/evidence/W7-07_pipeline_timing.md`를 함께 인용해 과장하지 않음.
- 추가: **증거 ①② 재현** 섹션 — 정의서 §3의 두 합격 기준을 재현 명령 + 실행 증거
  파일(`W5-09_gate_negative.md`/`ADR-0005_W5-06_revalidation.md`, `W6-06_retrain_evidence.md`
  + `W6-06_retrain_before_after.json`)로 표로 정리. 상단 네비게이션에도 앵커 추가.
- 수정: Quick Start 아래 "단일 명령으로 정리돼 있지 않습니다" NOTE를 위 Demo 섹션으로
  연결하도록 갱신(공백을 숨기지 않고 해결로 교체).

## 2. 데이터 카드 (`docs/data_card.md`) 최종화

- 헤더를 "초안(W3-08)" → "최종(W8-03)"으로 변경.
- **사용자 확인 필요 사항 발견**: §4가 스스로 "외부 배포본(GitHub Release/HF)이
  W3-12 버그 수정 전 OOD 데이터를 담고 있고, 이 카드가 최종본이 되기 전 반드시
  해결해야 한다"고 적어 둔 블로커가 있었음 — 외부 서비스 재업로드는 내가 대신
  결정·실행할 수 없는 행동이라 AskUserQuestion으로 확인. 1차 결정은 "직접
  재업로드"였으나, 이후 사용자가 **재업로드하지 않기로 최종 결정**(2026-11-14,
  "굳이 외부에서는 안 해도 될 것 같아"). §4를 "영구적 한계로 확정" — 외부 배포본은
  구버전으로 영구히 남는다는 경고를 최종 상태로 유지.
- 그 외 내용은 기존 초안이 이미 충실해(실측 수치·한계·검증 결과 모두 [확인] 태그)
  추가 수정 없음.

## 3. 모델 카드 (`docs/model_card.md`, 신규 생성)

- 레지스트리 실제 데이터(`registry/mgn_retrain_20261112T212130/card.json`,
  `registry/history.jsonl`)에서 현재 champion의 `dataset_hash`, `valid_range`(7개
  파라미터 min/max), `gate_result`, 승격 이력(ep2 → 재학습 교체)을 그대로 인용.
  `git_sha` 필드는 카드 자체가 "재학습 루프가 git 명령을 안 써서 비워둠"이라고 이미
  정직하게 기록해 둔 걸 그대로 가져옴(설계상 공백, 버그 아님).
- W8-01 벤치마크 표에서 골든10 A/B 비교 요약 재인용 + **FNO는 자동 C5 게이트·C6
  레지스트리 승격을 실제로 거친 적이 없다**는 사실을 `grep -i fno registry/*.jsonl
  registry/manifest.json` 실행 결과(0건)로 확인해 명시 — 팀 채점 노트북 평가와
  자동화된 승격 플로우를 혼동하지 않도록 구분.
- 한계는 새로 쓰지 않고 W8-02를 인용 + 모델 카드 관점에서 가장 중요한 3가지
  (보잉·언더컷 미측정, FNO OOD 없음, 지연 미실측)만 재강조.
- **금지 표현 검사 범위에 추가**: `scripts/check_forbidden_expressions.py`의
  `SCAN_FILES`에 `docs/model_card.md` 추가(W7-06 당시 "파일이 아직 없어 추가 예정"으로
  남겨 둔 항목 — 생성과 동시에 반영). 작성 중 실제로 금지 표현("실제 fab 데이터")을
  한 번 썼다가 검사가 바로 잡아내 수정함 — 검사가 실제로 작동함을 자체 확인.
  회귀 테스트 `test_docs_model_card_is_scanned` 추가.

## 재검증

```bash
$ python scripts/check_forbidden_expressions.py
금지 표현 검사 통과

$ python -m pytest tests/unit/test_forbidden_expressions.py -q
6 passed in 0.19s
```

## 남은 작업

- 없음 — 데이터 카드 §4는 "재업로드하지 않음"으로 최종 확정됨(2026-11-14).
