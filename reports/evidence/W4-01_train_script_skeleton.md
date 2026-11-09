# W4-01: 학습 스크립트 뼈대 (FR-06)

- 날짜: 2026-11-09
- 코드: `src/etchpipe/train/run_train.py`(신규), `configs/model/mgn_v1.yaml`(신규 설정 예시)
- 테스트: `tests/unit/test_run_train.py`(5건)

## 범위 [확인]

체크리스트 문구 그대로: "YAML 설정 한 파일, 시드·설정·코드 SHA·패키지 버전을 `runs/`에
저장". 실제 MeshGraphNet 학습 루프(체크포인트·`metrics.json`)는 범위 밖 — `init_run()`이
반환하는 `runs/<run_id>/` 디렉터리에 W4-02~04에서 이어서 쓴다(설계서 C3 출력 스펙과 일치).

- `set_seed`: `random`/`numpy`/`torch` 시드를 한 번에 고정.
- `git_sha`: 현재 커밋 SHA, 커밋 안 된 변경이 있으면 `-dirty` 접미사(재현성 경고).
- `package_versions`: `torch`/`torch_geometric`/`torch_scatter`/`nvidia-physicsnemo`/
  `numpy`/`scipy` 버전을 `importlib.metadata`로 조회, 미설치면 `"not installed"`.
- `init_run`: 위 셋 + YAML 설정을 `runs/<run_id>/{config.yaml, git_sha, packages.json}`로
  저장. `run_id`는 타임스탬프+짧은 난수 접미사(같은 마이크로초 충돌 방지).

## 새 의존성: PyYAML [확인, 사용자 확인 후 추가]

`scripts/gen_splits.py`에 "PyYAML에 의존하지 않도록 YAML을 직접 읽지 않는다(새 의존성
추가는 사용자 확인 필요)"는 선례가 있었으나, 학습 설정은 값이 자주 바뀌어 상수로
손 동기화하는 방식이 더 이상 맞지 않는다고 판단해 사용자에게 추가 여부를 확인받고
`requirements-train.txt`에 `PyYAML==6.0.3` 추가(2026-11-09).

## 실행 중 발견한 문제: Windows cp949 디코딩 [확인, 수정 완료]

`configs/model/mgn_v1.yaml`에 한글 주석이 있어 `pathlib.Path.read_text()`(인코딩
미지정)가 Windows에서 기본 로캘(cp949)로 읽으려다 `UnicodeDecodeError`로 실패했다.
`read_text(encoding="utf-8")`/`write_text(..., encoding="utf-8")`로 명시해 수정—
다른 기존 파일(`registry.py` 등)의 `read_text()`/`write_text()`는 내용이 전부 ASCII라
같은 문제가 없어 손대지 않음(범위 밖).

## 테스트 [확인]

```
PYTHONPATH=src python -m pytest tests/unit/test_run_train.py -v
5 passed
  test_set_seed_makes_random_draws_reproducible
  test_git_sha_matches_head
  test_package_versions_reports_installed_packages
  test_init_run_writes_config_git_sha_and_packages
  test_init_run_creates_distinct_run_dirs

PYTHONPATH=src python -m pytest tests/unit -q
86 passed (기존 81 + 신규 5)

python -m ruff check src/etchpipe/train/run_train.py tests/unit/test_run_train.py
All checks passed!
```

## 실행 확인(실제 설정 파일) [확인]

```
PYTHONPATH=src python -m etchpipe.train.run_train configs/model/mgn_v1.yaml
runs\20261109T014354191805_ead551
```

산출물 확인: `config.yaml`(로드한 설정 그대로 재직렬화), `git_sha`(40자+`-dirty`,
이 작업 자체가 아직 커밋 전이라 dirty는 정상), `packages.json`(호스트에는
`torch_geometric`/`torch_scatter`/`nvidia-physicsnemo`가 설치돼 있지 않아
`"not installed"` — 이 셋은 train 컨테이너 전용이라 호스트에 없는 게 정상, `numpy`/
`scipy`/`torch`는 버전 값이 정상 조회됨).

## 영향 범위

- `configs/model/mgn_v1.yaml`은 W4-02~04(학습 루프·과적합 테스트·본 학습)가 그대로 로드해
  쓸 수 있는 1차 초안이다. `processor_size=6`은 W4-04의 축소 모델 값에 맞춤 [제안].
  **정정(2026-11-09, 재정정)**: GPU 팀원 체크포인트의 `processor_layers`가 12개라 한 번
  `processor_size=12`로 바꿨었는데, W4-06에서 `load_state_dict`가 실패해 재확인 — physicsnemo
  소스상 `processor_layers` 개수는 `2*processor_size`(edge_block·node_block 번갈아 쌓임)라
  12개는 `processor_size=6`이 맞다[확인, `load_state_dict` 성공으로 검증]. 결국 이 파일의
  원래 `[제안]` 값(6)이 맞았다 — 상세는 `reports/evidence/W4-06_rollout_vs_baseline.md` 참고.
- `runs/`는 이미 `.gitignore`(`/runs/`)로 제외돼 있어 추가 조치 불필요.
