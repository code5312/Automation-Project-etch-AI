# W7-08: 문서·코드 불일치 목록 정리 후 문서 수정

정본 문서(`project-definition.md`, `project-design.md`)를 전부 읽고, 체크리스트·실제
코드·실제 산출물과 교차 검증해 불일치 후보 13건을 조사했다. 7건이 실제 불일치로
확인(DOC-02)돼 문서를 수정했고, 나머지는 이미 현실과 일치해 그대로 둔다.

## 확인된 불일치(수정함, DOC-02)

| # | 문서 위치 | 불일치 내용 | 근거 | 조치 |
|---|---|---|---|---|
| 1 | definition.md §10 확인됨 | "학습 1스텝 약 40~180ms(노드 300~400개)" — 1주차 토이 규모 추정이 실제 champion급(tuned) MeshGraphNet 성능으로 오인될 수 있었음 | 4주차 역할분담 메모(`project-checklist.md`, 150~290초/에폭÷71전이쌍≈2.1~4.1초); W7-07 격리 재실측(`reports/evidence/W7-07_pipeline_timing.md`, 68~87초/에폭÷38전이쌍≈1.8~2.3초) | 옛 수치 폐기 명시 + 실제 범위로 교체(definition.md §10), design.md §14 동일 문구 교체 |
| 2 | definition.md §10 미확인 | "다단계 롤아웃 — 가장 불확실" | W4-05~07 `[x]` 완료, `reports/evidence/W4-05_rollout_v1.md`, D4 게이트 통과 | 확인됨으로 이동 |
| 3 | definition.md §10 미확인 | "FNO 구현·학습, 추론 속도, 속도 이득" | W6-11 `[x]`(2026-11-12), `runs/fno_tuned/model.pt` 등 실존, 골든10 Chamfer 0.069µm 합격 | 확인됨으로 이동 |
| 4 | definition.md §10 미확인 | "학습·시뮬레이션 환경 공존 여부(기본은 분리)" | `src/etchpipe/retrain/execute.py` 모듈 docstring이 이미 "etch-serve:0.1이 환경 3분리에서 벗어난 예외"라고 명시 | 확인됨으로 이동 + 예외 조건 기록 |
| 5 | definition.md §9 | "추론 속도 비(ViennaPS 시간÷모델 시간) [ADR-0005]"가 구현·보고된 것으로 오인될 수 있었음 | `eval/metrics.py::speed_ratio` 구현·단위테스트는 있으나(`tests/unit/test_metrics.py`) 실제 데이터로 호출·보고된 적 없음(grep으로 호출부 없음 확인) | §10 미확인에 "구현됨, 실측 보고 없음"으로 명확화 |
| 6 | design.md §5, §7 예시 | `mask_width`/`mask_height`를 categorical choices(`"narrow"/"mid"/"wide"`, `"low"/"high"`, 요청 예시의 `"mid"`/`"low"`)로 적었으나 실제 구현(`src/etchpipe/serve/app.py::Recipe`, definition.md §7.1)은 연속 float(µm) | `app.py` Recipe 필드: `mask_width: float = Field(gt=0, le=25.0, ...)` 등 | YAML/JSON 예시를 연속값으로 교체 + 수정 각주 |
| 7 | design.md §7 API 표 | 실제 구현된 `POST /api/predict`, `GET /api/models/compare`, `GET /api/gate/history`(대시보드 C9용, W7-01에서 추가)가 표에 없음 | `src/etchpipe/serve/app.py` 라우트 정의(`@app.post("/api/predict")` 등) | 표에 3개 행 추가 |

## 조사했으나 이미 일치(수정 안 함)

| # | 후보 | 결론 | 근거 |
|---|---|---|---|
| 8 | definition.md §5.1 "대시보드 화면 3개" | 일치 | W7-01 `[x]`(2026-11-14) 완료 문구가 화면 이름까지 정확히 동일 |
| 9 | definition.md §13 "쿠버네티스 kind 1회 성공 목표" | 일치 | W1-S4 `[x]`: kind 클러스터, 파드 4개 Completed, 재생성+skip 확인 — 목표한 경로로 실제 성공(`xargs -P` 대체 경로로 간 적 없음) |
| 10 | definition.md §7.3 규모·분할(train 300/val 40/test 40/OOD 20) | 일치 | `data/splits.json` 실제 카운트: train=300, val=40, test=40, ood=20 — 정확히 일치 |
| 11 | definition.md §7.3 golden=10 | 일치 | `golden/golden.json`의 `run_ids`/`selected` 길이 10 (최초 `len(dict)`로 5가 나와 불일치로 오판했으나, dict 최상위 키가 5개(`rule`/`source_split`/`run_ids`/`selected`/`golden_hash`)였을 뿐 — 재확인으로 정정) |
| 12 | design.md §7 `/predict` 응답 형태(`model_version` + `results` 배열) | 일치 | `app.py::PredictResponse(model_version: str, results: list[PredictResult])` 그대로 |
| 13 | checklist.md:133 OOD +100건 확장 보류 | 이미 추적 중인 별개 항목(DOC- 필요로 명시돼 사용자 확인 대기) — 이번 조사로 새로 발견한 게 아니라 기존 오픈 아이템이라 W7-08 범위에서 중복 처리 안 함 | `project-checklist.md:133` |

## 수정한 파일

- `docs/project-definition.md` §9, §10 (확인됨/미확인 섹션 재구성)
- `docs/project-design.md` §5(YAML 예시), §7(JSON 예시 + API 표), §14(스파이크 비고)

## 미확인/남은 작업

- #5(추론 속도 비)는 "구현은 있으나 미보고" 상태로만 문서화했다 — 실제로 ViennaPS
  시간÷모델 시간을 계산해 보고하는 것은 이번 항목 범위 밖(필요하면 별도 체크리스트
  항목으로).
- #13(OOD +100건)은 기존에 이미 열려 있던 결정 대기 항목이라 이번에 손대지 않음 — 여전히
  사용자 확인 필요 상태.
