# W7-06: 금지 표현 검사 (코드·README·대시보드·모델 카드, 정의서 6장 기준 → FR-18)

## 기존 검사 재확인
```
PYTHONUTF8=1 python scripts/check_forbidden_expressions.py
-> 금지 표현 검사 통과 (exit 0)
```
`SCAN_DIRS = ["src", "scripts", "tests"]` + `SCAN_FILES = ["README.md"]` 기준 통과.
대시보드 정적 파일(`src/etchpipe/dashboard/static/*.js`)은 `src/` 하위라 이미 스캔 범위에 포함됨.

## 발견한 범위 공백
체크리스트 항목이 "코드·README·대시보드·모델 카드"를 명시하는데, 스크립트는
`docs/` 전체를 스캔 대상에서 뺀다. 이유는 `CLAUDE.md`·`docs/project-definition.md` §6이
금지 표현 자체를 "정의"하기 위해 인용하기 때문(자기 자신을 위반으로 잡는 문제)인데,
이 블랭킷 제외가 `docs/data_card.md`(실제 데이터 문서)와 앞으로 생길
`docs/model_card.md`(W8-03)까지 같이 빠지게 만든다 — 체크리스트가 명시한 대상이 실제로는
검사 안 되고 있었음.

### `docs/*.md` 전체에 금지 표현 5개를 직접 grep한 결과
- `docs/adr/0003-signed-sdf-convention.md:78` — "실시간" 매칭, 실제로는 무관한 일반 단어 용법
  (오탐).
- `docs/project-definition.md:55,67` 등 — 금지 표현 "정의" 문장 자체(규칙을 나열하는 텍스트).
- `docs/project-design.md:30,329` — "~을 대체한다"(낡은 방법을 새 방법으로 교체한다는 일반
  의미)와 §5.2 선택 범위 설명 안의 "실시간"(제외 범위 설명, 금지 주장 아님). 둘 다 오탐.
- `docs/data_card.md` — 전체 그렙 결과 **매칭 없음**(금지 표현 0건, 깨끗함).

결론: `project-definition.md`/`project-design.md`/`adr/*`는 규칙 정의·일반 단어 용법이 많아
그대로 두면 오탐만 generate하므로 계속 제외. 반면 `data_card.md`는 실제 내용 문서이고
현재 깨끗하므로 스캔 대상에 추가해도 안전함.

## 조치
`scripts/check_forbidden_expressions.py`의 `SCAN_FILES`에 `"docs/data_card.md"` 추가.
`docs/model_card.md`는 아직 파일이 없음(W8-03에 생성 예정) — 생성 시 같은 목록에 추가하라는
주석을 남겨둠.

```
find docs -iname "*model_card*" -o -iname "*modelcard*"
-> (결과 없음, 2026-11-14 확인)
```

재검사:
```
PYTHONUTF8=1 python scripts/check_forbidden_expressions.py
-> 금지 표현 검사 통과 (exit 0)
```

## 테스트
`tests/unit/test_forbidden_expressions.py`에 `test_docs_data_card_is_scanned` 추가
(`docs/data_card.md`에 금지 표현을 심으면 실제로 잡히는지 확인 — 스캔 범위 확장이
동작하는지에 대한 회귀 테스트).
```
PYTHONUTF8=1 python -m pytest tests/unit/test_forbidden_expressions.py -q
-> 5 passed in 0.18s
```

## 미확인 / 남은 작업
- `docs/model_card.md`(W8-03 생성 예정)는 생성 즉시 `SCAN_FILES`에 추가해야 함 — 지금은
  파일이 없어 대상에 없음(스크립트가 `p.is_file()`로 존재 여부를 확인하므로 없는 파일을
  미리 넣어도 에러는 안 나지만, 당장은 의미가 없어 추가하지 않음).
- `docs/project-definition.md`/`project-design.md`/`adr/*`는 규칙 정의·일반 단어 용법 때문에
  계속 제외 — 매처가 단순 부분 문자열 비교라 정교한 문맥 구분은 못 함, 현재로서는 범위
  설계로 대응.
