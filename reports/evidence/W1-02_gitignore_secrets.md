# W1-02 증거: .gitignore·시크릿 스캔

- 날짜: 2026-10-27

## .gitignore

체크리스트 규칙대로 `data/`, `runs/`, `registry/`, `reports/*`(`reports/evidence/`는 예외), `*.vtp`(`golden/`, `tests/fixtures/`는 예외)를 제외. 추가로 체크포인트(`*.pt`/`*.pth`/`*.ckpt`), Python 캐시, 다운로드한 `kind.exe` 바이너리를 제외.

## 시크릿 스캔(pre-commit)

`.pre-commit-config.yaml`: `Yelp/detect-secrets@v1.5.0` + `pre-commit/pre-commit-hooks@v6.0.0`(`detect-private-key`). 둘 다 실제 GitHub 태그로 존재하는지 GitHub API로 확인(`curl https://api.github.com/repos/Yelp/detect-secrets/tags`, `.../pre-commit-hooks/tags`) [확인].

베이스라인 생성(python:3.11-slim 컨테이너, `detect-secrets==1.5.0`):

```
detect-secrets scan --all-files --exclude-files '\.git/' > .secrets.baseline
```

1차 실행에서 `.pytest_cache/CACHEDIR.TAG`, `.ruff_cache/CACHEDIR.TAG`(둘 다 gitignore 대상 캐시 파일)가 "Hex High Entropy String"으로 오탐됨 → 두 캐시 디렉터리를 삭제하고 재실행하여 결과:

```
files with results: 0
```

→ 현재 저장소 파일 전체에서 탐지되는 시크릿 없음 [확인]. `.secrets.baseline`을 커밋 대상에 포함.

## 미확인

- `pre-commit` 자체(훅 러너)는 설치·실행하지 않았다(`detect-secrets scan`으로 베이스라인만 직접 생성·검증). `pre-commit install` 후 실제 `git commit` 훅 트리거는 저장소에 첫 커밋이 없어 아직 확인하지 못함.
