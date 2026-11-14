# W8-09: GPU 환경 실행 가능성 (v1.0 이후 추가 조사)

v1.0 태그 이후 팀원이 실제 GPU(NVIDIA RTX A5000)에서 W6-06 재학습 시나리오를 그대로
재현해 전달한 결과를 바탕으로, "GPU가 있으면 무엇을 그대로 쓸 수 있고 무엇은 아직
안 되는지"를 코드 기준으로 정리한다.

## 1. 지금 코드에서 이미 되는 것 [확인]

`src/etchpipe/train/train_main.py`(본 학습 진입점, `python -m etchpipe.train ...`)는
처음부터(W4-04) `--device cuda`를 지원하도록 설계돼 있다 — `model.to(device)`,
`train_one_epoch(..., device=device)`, `evaluate(..., device=device)`가 전부
`device` 인자를 그대로 받는다(`train_main.py:40,55,75,81,108,138,157`). 즉 **단일
모델을 처음부터 학습하는 경로는 지금 그대로 GPU 환경에서 돌릴 수 있다**:

```bash
python -m etchpipe.train <config.yaml> --device cuda
```

## 2. 지금 코드에서 아직 안 되는 것 [확인]

**자동 재학습 루프(C8, `python -m etchpipe.retrain run`)는 GPU를 못 쓴다.**
`src/etchpipe/retrain/execute.py::make_fine_tune_fn`의 `fine_tune_fn`이
`train_one_epoch(..., device="cpu", ...)`로 **하드코딩**돼 있다(`execute.py:130`) —
`device` 인자 자체가 없다. 즉 이번에 재학습 루프를 GPU로 돌리고 싶어도 지금 코드로는
불가능하고, `fine_tune_fn`에 `device` 파라미터를 추가하고 호출부(`run()`,
`run_retrain_demo.py`)까지 바꿔야 한다 — **아직 안 한 작업**이다.

팀원이 전달한 GPU 실행은 이 자동 루프(`python -m etchpipe.retrain run`)를 그대로 쓴
게 아니라, 같은 시작 체크포인트·같은 데이터·같은 학습 설정(3에폭, Adam lr=1e-4)을
**별도 하네스(외부 GPU Pod)로 재현**한 것이다 — 저장소의 어떤 명령 한 줄로 재현되는
경로가 아니므로, README/체크리스트에 "이 명령으로 GPU 재학습을 돌리세요"라고 적지
않는다(실행해 보지 않은 것을 사실로 쓰지 않는다, CLAUDE.md 4항).

## 3. 팀원 실측 — 그 수정을 했을 때 기대할 수 있는 효과의 증거 [확인]

같은 `dataset_hash`(`26a29dcca7da`)·같은 시작 체크포인트(`mgn_tuned_ep2`)·같은 설정
(3에폭, Adam lr=1e-4)으로, CPU 대비 GPU 결과:

| | CPU(참조 재측정) | GPU(RTX A5000) |
|---|---|---|
| 3에폭 학습 합계 | 7,425초(2시간 4분) | 79.89초 |
| 배율 | — | **약 92.9배** |
| Chamfer 평균(20개 평가) | 0.535(미세조정 전) → 0.498(우리 실제 CPU champion, 공개값) | 0.462(GPU 후보) |
| 깊이오차 평균 | 7.05%(미세조정 전) → 4.95%(우리 실제 CPU champion) | 4.63%(GPU 후보) |

품질은 떨어지지 않으면서 학습 시간만 크게 줄었다 — W7-07에서 찾은 "fine-tune이 전체
파이프라인 시간의 92%" 병목을, GPU를 쓸 수 있게 되면 거의 해소할 수 있다는 실측
근거다. 단, 이 GPU 후보는 레시피 20개짜리 간이 평가(ADR-0004류 임계값)만 거쳤고,
현재 운영 중인 ADR-0005(레시피별 통과율 90%+윌콕슨 퇴보 검정) 전체 판정과 val-40
전체는 아직 돌리지 않았다 — "우리 실제 게이트를 통과했다"가 아니라 "품질이 나빠지지
않으면서 훨씬 빠르다는 예비 실험"으로 읽어야 한다.

## 4. 향후 과제로 추가

`reports/evidence/W8-07_overclaim_check_and_future_work.md`의 "#5 GPU 기종 기록"
항목을 아래로 갱신한다: GPU 기종(RTX A5000)과 실측 배율(92.9배)은 이제 기록됐으나,
**자동 재학습 루프 자체의 GPU 지원(`execute.py::make_fine_tune_fn`에 `device` 파라미터
추가)은 여전히 미구현** — 다음에 손댈 구체적인 지점까지 적어 둔다.

## 미확인/남은 작업

- `retrain/execute.py`의 `device` 파라미터 추가 자체는 하지 않음(이번은 조사·기록만).
- 이 GPU 후보 체크포인트에 대한 ADR-0005 전체 판정, val-40 전체 평가는 미실행.
