# W7-04: 보안 점검

의존성 취약점 스캔, 시크릿 스캔, `torch.load weights_only`, 커밋 이력 확인 4개를
전부 실제 도구/실측으로 확인했다.

## 1. 의존성 취약점 스캔 (`pip-audit`)

`requirements-dev.txt`/`requirements-sim.txt`는 `pip-audit -r`로 직접 스캔(취약점
없음). `requirements-train.txt`/`requirements-serve.txt`는 `torch_scatter`의 빌드가
torch를 요구해(`--no-build-isolation`, 이 프로젝트 자체가 문서화한 설치 순서 제약)
`pip-audit`의 격리 해석 모드가 못 돈다 — 대신 **실제 설치된 `etch-train:0.1`/
`etch-serve:0.1` 컨테이너 안에서** `pip-audit`(의존성 없이, 설치된 패키지 직접 스캔)
실행.

**발견(수정 전)**: 두 이미지 모두 `pip==24.0`(CVE 6건, 수정 버전 25.3~26.2)·
`setuptools==79.0.1`(CVE 2건, 수정 버전 83.0.0/84.1.0) — base 이미지(`python:3.11-slim`)
가 가져온 버전이고 `requirements-*.txt`엔 없음. `etch-sim:0.1`은 base 이미지가 더 최신
(pip 26.2.2)이라 깨끗함. `torch`/`torchvision`(CPU 빌드, `+cpu` 접미사)은 PyPI 인덱스
매칭 실패로 스캔 자체가 안 됨(도구 한계, 미확인으로 남김).

**조치(사용자 승인)**: `docker/Dockerfile.train`·`docker/Dockerfile.serve` 맨 뒤에
`pip install --upgrade pip setuptools` 레이어 추가 — 기존 torch/physicsnemo 캐시를
안 깨도록 끝에 둠. 재빌드 후 재스캔: **두 이미지 다 "No known vulnerabilities found"**
확인. 재빌드 후 전체 테스트(`etch-train:0.1` 215 passed, `etch-serve:0.1` test_serve.py
34 passed) 재확인 — 회귀 없음.

## 2. 시크릿 스캔 (`detect-secrets`)

`detect-secrets scan --baseline .secrets.baseline` 실행 — **신규 발견 4건**, 전부
`golden/raw/r*/meta.json`의 `params_hash` 필드("Hex High Entropy String" 오탐, 10자리
hex 콘텐츠 해시를 엔트로피 휴리스틱이 잠재적 비밀로 플래그). 실제 값 확인(예:
`golden/raw/r0007/meta.json` 22행 `"params_hash": "8c3c27d1a6"`) — README가 이미
문서화한 콘텐츠 해시(`sha1(params)[:10]`)이지 자격증명이 아님. `is_secret: false`로
감사(audit) 처리해 베이스라인에 기록.

`.pre-commit-config.yaml`의 `detect-private-key` 훅에 대응해 저장소 전체에서
`BEGIN ... PRIVATE KEY`/`BEGIN RSA`/`BEGIN OPENSSH` 패턴 직접 검색 — 없음.

## 3. `torch.load` `weights_only` 적용 확인

전체 코드베이스에서 `torch.load(` 호출 4곳 확인:
- `tests/unit/test_train_main.py`(2곳): 이미 `weights_only=True` — 양호.
- `src/etchpipe/eval/eval.py::load_model_a`, `scripts/run_rollout_comparison.py::
  _load_model_a`: **`weights_only=False`였음**(임의 pickle 역직렬화 허용 — 악성 .pt가
  섞이면 코드 실행 위험). `load_model_a`는 서빙(`inference.py::load_champion`)이 실제
  champion을 로드할 때 쓰는 바로 그 함수라 실제 공격 표면과 직결.

**수정**: 둘 다 `weights_only=True`로 변경 + `torch.serialization.add_safe_globals`로
필요한 numpy 타입 4개(`numpy._core.multiarray._reconstruct`, `numpy.ndarray`,
`numpy.dtype`, `numpy.dtypes.Float32DType`)만 명시적으로 허용 — 단순히
`weights_only=False`로 되돌리는 것보다 안전(임의 객체가 아니라 이 4개로 좁힘). 체크포인트
안의 `mu_n`/`sd_n`/`mu_e`/`sd_e`가 (구버전 포맷에서) numpy 배열이라 기본 allowlist만으로는
로드가 깨져서[확인, `runs/mgn_tuned/model.pt`로 직접 재현] 이 4개가 필요했다. 실제
champion(`runs/mgn_tuned/model.pt`, numpy 포맷)과 재학습 산출물
(`runs/retrain/mgn_retrain_20261112T212130/model.pt`, list 포맷) 둘 다 실제로
`weights_only=True`로 정상 로드됨을 `etch-train:0.1`에서 직접 확인.

## 4. 커밋 이력에 데이터·키 없음 확인

`git log --all`(85개 커밋) 기준:
- `data/*` 경로를 추가한 커밋 **0건** — 한 번도 커밋된 적 없음(확인).
- `.env`/`.pem`/`.key`/credential류 파일을 추가한 커밋 **0건** — grep 매칭은 전부
  `.secrets.baseline`·평가 문서 파일명 자체(파일명에 "secret"이 들어간 합법적 파일)였음.
- 전체 히스토리 diff에서 `ETCH_API_KEY=`/`api_key`/`AKIA...`/`BEGIN ... KEY` 패턴
  검색 — 실제 값 매칭 0건(전부 코드 식별자·README 문서 텍스트). 의심스러워 보였던
  거대한 base64 블록 1건은 추적 결과 `golden/raw/r0187/snapshots/*.vtp`의 정상적인
  바이너리 콘텐츠(VTK 내부 압축 배열이 XML 안에 base64로 들어있어 git이 텍스트로
  diff를 냄) — `golden/`은 의도적으로 커밋하기로 한 예외 데이터라 문제 없음.

## 검증 요약
```
pip-audit (dev/sim, -r 직접): 취약점 없음
pip-audit (train/serve, 컨테이너 내부 설치 스캔, 수정 전): pip/setuptools CVE 15건
Dockerfile.train/serve에 pip/setuptools 업그레이드 레이어 추가 -> 재빌드 -> 재스캔:
  두 이미지 다 "No known vulnerabilities found"
etch-train:0.1 전체 테스트: 215 passed (재빌드 후 회귀 없음)
etch-serve:0.1 test_serve.py: 34 passed (재빌드 후 회귀 없음)
detect-secrets: 신규 4건 전부 오탐(params_hash) 확인·audit 처리
load_model_a 실제 체크포인트 2종(numpy/list 포맷) weights_only=True 로드 성공 확인
git log --all: data/·키 파일 커밋 이력 0건
```

## 미확인 / 남은 작업
- `torch`/`torchvision`(CPU 빌드)은 `pip-audit`가 PyPI 인덱스에서 매칭 못 해 취약점
  스캔 자체가 안 됨 — 다른 도구(예: OSV API 직접 조회)로 보강 필요할 수 있음.
- `golden/`에 커밋된 `.vtp` 바이너리가 git 저장소 크기에 미치는 영향은 범위 밖으로 둠
  (보안이 아니라 용량 이슈).
