# W6-11: FNO 구현(모델 B 래퍼) + SDF 0등고선 복원

- 날짜: 2026-11-12
- 코드: `src/etchpipe/models/fno.py`(신설), `requirements-train.txt`/`docker/Dockerfile.train`
  (scikit-image 추가), `tests/unit/test_fno.py`(신설)

## 결정: 재구현하지 않고 GPU 팀원 전달물을 그대로 감싼다 [확인]

`runs/fno_tuned/`에 GPU 팀원이 이미 학습·검증한 FNO 체크포인트(`model.safetensors`)와
완결된 추론 코드(`inference.py`/`preprocess.py`)가 있다(README: 골든 10 Chamfer
0.069µm, 깊이오차 0.35%, 합격). SDF 내부 경계 필터링처럼 미묘하게 틀리기 쉬운 부분이
이미 검증돼 있어서, `etl.py`의 자체 SDF 구현(부호 규약은 같지만 필터링 알고리즘이
다름 — 걷기 기반 vs y_below 컷오프)으로 재구현하는 대신 그대로 재사용했다.

## 사전 검증 [확인, 실측]

```
cd runs/fno_tuned && python inference.py
-> final depth true 33.22 um / pred 33.05 um   (README 문서화 값과 일치)
```

`scikit-image`가 없어서(`phi_to_curve`의 `find_contours`용) 추가 설치 필요함을 발견 —
버전 고정 없이 설치해 실제 해석된 `0.26.0`을 그대로 핀(requirements-train.txt,
docker/Dockerfile.train, CI `unit-tests`/`gate-negative` job).

## 래퍼 [확인] — `FNOModel`

- `sdf_from_vtp(vtp_path)`: `.vtp` → 학습 때와 같은 SDF 격자(128×256µm, teammate의
  `preprocess.vtp_to_sdf` 그대로).
- `rollout_sdf(sdf0, params, n_steps)`: `inference.rollout` 그대로.
- `curve_from_sdf(sdf)`: `inference.phi_to_curve` 그대로 — **SDF 0등고선 복원**
  (`skimage.measure.find_contours`), 왼쪽→오른쪽 (N,2) µm.
- `predict_curve(vtp_path, params, n_steps)`: 위 셋을 묶어 모델 A와 같은 출력 형태
  (최종 표면선 점 배열)로 — W6-12 비교표에서 같은 Chamfer/깊이 지표로 바로 비교 가능.

physicsnemo/scikit-image는 함수 안에서 지연 임포트(`_ensure_loaded`) — 없는 환경에서도
클래스 자체는 import 가능(`eval.load_model_a`와 같은 패턴).

## 테스트 [확인, 실측 — 전부 실제 체크포인트·실제 데이터]

```
PYTHONPATH=src python -m pytest tests/unit/test_fno.py -v
  test_load_fno_config_matches_known_grid PASSED
  test_rollout_and_curve_match_documented_example_depth PASSED   # 33.22/33.05 재확인
  test_predict_curve_from_vtp_returns_point_array PASSED         # golden r0007 t000.vtp 사용
→ 3 passed
python -m ruff check src/etchpipe/models/fno.py tests/unit/test_fno.py → All checks passed
```

`runs/fno_tuned/`는 `.gitignore` 대상(로컬 전용, `mgn_tuned/model.pt`만 예외)이라
CI에서는 `pytest.mark.skipif`로 건너뜀 — `run_sim.py`/`run_drift_sim.py`와 같은 이유로
로컬 실측 전용.

## 알려진 한계

- **GPU 팀원 쪽**: 체크리스트 원래 계획은 "CPU 학습(격자 128×256)"였으나 실제로는
  GPU로 학습·전달됨([확인], README에 기록된 사용자 결정) — 실제 학습 자체는 이 세션
  범위 밖(GPU 팀원 수행).
- **모델 A/B 비교(W6-12)는 별도**: 이 항목은 래퍼·I/O 매핑까지만, 실제 같은
  test-split 비교표는 다음 항목.
