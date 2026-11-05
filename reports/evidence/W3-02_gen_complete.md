# W3-02: 본 데이터 생성 완료(kind Indexed Job, N=400)

- 날짜: 2026-11-05
- 생성 시작: 2026-11-03T01:38:17Z, 완료: 2026-11-05T07:14:01Z — **약 53.6시간(2.23일)**,
  `W3-01_probe8_depthmax_kind.md`의 parallelism=2 추정(약 2.6일)과 일치.

## Job 상태 [확인]

```
kubectl --context kind-etch-pipeline get job etch-sim-gen -o jsonpath='{.status}'
{"completedIndexes":"0-39","completionTime":"2026-11-05T07:14:01Z",
 "conditions":[...,{"reason":"CompletionsReached","status":"True","type":"Complete"}],
 "succeeded":40,"startTime":"2026-11-03T01:38:17Z"}
```

40/40 shard(Indexed Job) 전부 `Succeeded`. `status.failed` 필드 없음(=0건, 재시도 발생하지 않음).

## 데이터 완전성 [확인]

PVC(`etch-data`) `/data/raw`에 정확히 400개 run 디렉터리 존재(`ls /data/raw | wc -l` → 400).
로컬 백업(`data/raw/`, PVC와 diff 0건 확인 후 백업)에서 400건 전부 `meta.json` 존재·JSON
파싱 성공 확인(`python` 스크립트로 전수 검사, 누락 0건·파싱 실패 0건).

## 백업/배포

- 로컬: `data/raw/`(81MB), `data/recipes.json` — PVC와 diff 0건 확인 후 `kubectl cp`로 백업.
- GitHub Release: https://github.com/code5312/Automation-Project-etch-AI/releases/tag/data-v1-n400
  (`etch-raw-n400.zip`, 40.1MB, 400 run + `recipes.json`, 스파이크 잔재 13건 제외 — git 커밋 아님,
  CLAUDE.md "데이터 커밋 금지"와 충돌 없음)
- Hugging Face: https://huggingface.co/datasets/code5312/etch-ai-automation-data (동일 zip + Dataset Card)

## 실패율 요약 스크립트 관련 [미확인 — 후속 필요]

`scripts/sim_progress.py`는 `status.phase=Running` 파드를 전제로 동작해서, Job이 **완전히 끝난
뒤**에는 Running 파드가 없어 `kubectl get pods ... -o jsonpath={.items[0]...}`가 빈 배열 인덱싱으로
에러를 낸다(이번에 실제로 재현해 확인). 지금은 위 섹션처럼 `kubectl get job`/`du`/직접 파이썬 검사로
대체했음 — 완료 후 상태까지 처리하는 개선은 범위 밖으로 남겨둠(필요 시 별도 항목).

## 남은 것

- 스파이크 잔재 13건(`r0000_s1k2` 등, W1-S1/S2/S3)은 로컬 `data/raw/`에 그대로 남아 있음 — 본 데이터
  400건과 이름이 겹치지 않아 혼선 없음, 별도 정리 불필요로 판단.
