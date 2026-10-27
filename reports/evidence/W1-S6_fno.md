# W1-S6: FNO 모듈 경로·시그니처 확인, forward/backward, 1에폭 시간

- 목표: `nvidia-physicsnemo==2.2.2`에 FNO가 실제로 어디 있는지, 생성자 시그니처가 무엇인지 확인하고 128×256 입력으로 forward/backward 1회 + CPU 시간 측정
- 시간 상자: 1시간
- 데이터 의존성 없음(S1과 무관) — S1과 동시에 진행

## 실행

```
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install nvidia-physicsnemo==2.2.2
python -c "import inspect; from physicsnemo.models.fno import FNO; print(inspect.signature(FNO.__init__))"
```

모듈 위치: `physicsnemo/models/fno/fno.py` (클래스 `FNO`, import 경로 `physicsnemo.models.fno.FNO`).

시그니처(실제 설치본에서 확인):
```
(self, in_channels: int, out_channels: int, decoder_layers: int = 1, decoder_layer_size: int = 32,
 decoder_activation_fn: str = 'silu', dimension: int = 2, latent_channels: int = 32,
 num_fno_layers: int = 4, num_fno_modes: Union[int, List[int]] = 16, padding: int = 8,
 padding_type: str = 'constant', activation_fn: str = 'gelu', coord_features: bool = True) -> None
```

## 중요한 발견: torchvision 버전 미지정 시 import 자체가 깨짐

`torch==2.14.1`만 깔고(`torchvision` 버전 지정 없이) `from physicsnemo.models.fno import FNO`를 실행하면:

```
RuntimeError: operator torchvision::nms does not exist
```

`physicsnemo.models.__init__`가 FNO와 무관한 DiT 모델을 먼저 import하면서 `timm`→`torchvision`을 거치는데, pip가 자동으로 고른 torchvision이 `torch==2.14.1`과 버전이 안 맞아 깨진다. `requirements-train.txt`에 이미 명시해 둔 `torchvision==0.29.1`(W1-03에서 확인된 조합)을 명시적으로 같이 설치하면 해결된다 — **`torchvision`을 빼면 FNO조차 못 쓴다는 뜻이므로 requirements-train.txt에 그대로 유지해야 한다**는 근거가 하나 더 생겼다.

## 측정값 (128×256 입력, CPU, 4 스레드, batch=1)

```
params(M): 2.365
forward_sec: 0.975
backward_sec: 0.662
10step_sec: 6.471 (per_step_ms: 647.1)
```

- 파라미터 수 약 236만 개(in/out channel=1, latent=32, fno_layers=4, modes=12).
- forward 1회 약 1초, backward 포함 1 step 약 0.65~1초 → 1 "에폭"(여기선 같은 입력 10 step 반복으로 근사, 실데이터 배치 없음) 약 6.5초.
- **실데이터(진짜 배치 크기·채널 수)가 아직 없어 이 수치는 근사치** — W6-11(FNO 구현) 시점에 실제 거리장 11채널·배치 크기로 재측정 필요 [미확인: 실제 입력 기준 수치].

## 판정

**통과(진행 가능)**: FNO 모듈 위치·시그니처 확인됨, CPU forward/backward 정상 동작. 단, torchvision 핀이 없으면 즉시 깨지므로 requirements-train.txt의 명시적 pin을 반드시 유지해야 한다는 제약을 기록.

## 대안

불필요(통과). 참고용으로 남길 위험: physicsnemo의 모델 패키지 `__init__`가 무관한 모델(DiT 등)까지 전부 import하므로, 의존성 중 하나(예: timm)가 깨지면 FNO만 쓰려 해도 전체가 막힌다 — 추후 의존성 업그레이드 시 주의.
