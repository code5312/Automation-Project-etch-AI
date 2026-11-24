# W9-02: 데이터 버전·백업 정책 ADR

## 1. 문제

`docs/project-checklist.md` W8-03에서 데이터 카드 최종화 중 "외부 배포본
(GitHub Release `data-v1-n400`, Hugging Face `code5312/etch-ai-automation-data`)을
W3-12 버그 수정본으로 재업로드할지"를 처음 공식적으로 물었다. 1차 결정은 "직접
재업로드"였으나 같은 날 사용자가 반전해 "재업로드하지 않음"으로 최종 확정
(2026-11-14, `reports/evidence/W8-03_readme_cards.md`). 두 결정 사이에 명시적 기준이
없었다 — 다음에 비슷한 데이터 버그가 나오면 같은 논의를 원점에서 다시 해야 한다.

## 2. 조사

- `docs/project-checklist.md:656-669`(W8-03, W9-02 항목), `docs/data_card.md:87-92`
  (§4 "외부 배포본이 구버전" 한계 서술), `reports/evidence/W8-03_readme_cards.md`
  (실제 결정 반전 경위) 재확인.
- W3-11(ETL 버그, raw 미영향)과 W3-12(raw 자체 버그, 재생성·로컬 백업 완료)의 차이가
  "재업로드 논의가 실제로 벌어졌는지 여부"와 일치함을 확인 — 트리거 기준 설계의
  근거로 사용.

## 3. 결정

`docs/adr/0007-data-version-backup-policy.md` 작성. 핵심:
1. 재업로드 검토 트리거 3종(물리적 무결성 버그·split 전체 영향·게이트 판정 영향) —
   해당 시에만 AskUserQuestion으로 제안, 미해당이면 기본값은 "제안 안 함".
2. 외부 태그는 불변 — 재업로드 결정 시 새 버전 태그로 추가 배포(소급 수정 금지).
3. raw 재생성 시 로컬 백업 — W3-12에서 실제로 한 패턴을 일반 규칙으로 승격.
4. 실행 주체는 항상 사용자 — Claude는 트리거 판단·제안까지만, 외부 서비스 쓰기
   액션은 대신 실행하지 않음(CLAUDE.md 금지 목록과 같은 취지).

`docs/data_card.md` §4에 이 ADR을 가리키는 한 줄 추가(기존 "재업로드 안 함" 결정
서술은 그대로 유지 — 소급 변경 아님).

## 4. 검증

코드 변경 없음(정책 문서 ADR) — 테스트/CI 대상 없음. `scripts/check_forbidden_expressions.py`
스캔 대상에 `docs/adr/`는 포함되지 않음(기존 범위 유지, 확인):

```
$ python scripts/check_forbidden_expressions.py
금지 표현 검사 통과
```

## 5. 남은 사항

- 이 ADR의 트리거 기준은 아직 실제 다음 사례로 검증된 바 없음[제안] — 다음 데이터
  버그가 발생했을 때 기준이 실제로 유용한 1차 판단을 제공하는지 재확인 필요.
- **승인 완료(2026-11-24, 사용자 확인)** — ADR 상태를 "승인"으로 갱신, 체크리스트
  W9-02 완료 처리.
