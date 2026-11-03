"""W3-02: 본 데이터 생성 진행률·실패율 요약 + 실패 건 재시도.

PVC(etch-data)는 kind 노드 내부에 있어 호스트에서 직접 파일로 못 읽는다 — 현재 실행 중인
etch-sim-gen 파드에 kubectl exec로 들어가 /data를 조회한다. Job이 다 끝나 파드가 하나도
안 남았으면(Completed 파드는 기본적으로 삭제되지 않지만 모두 끝난 뒤일 수도 있음) --pod로
수동 지정한다.

사용법:
  python scripts/sim_progress.py                 # 진행률·실패율만 출력
  python scripts/sim_progress.py --retry          # Job이 Failed 상태면 delete+재적용
                                                   # (멱등, 완료분은 meta.json으로 자동 skip)
"""

import argparse
import contextlib
import json
import subprocess
import sys

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 기본 코드페이지에서 한글 출력 보정

CONTEXT = "kind-etch-pipeline"
JOB_NAME = "etch-sim-gen"
JOB_LABEL = f"job-name={JOB_NAME}"
JOB_YAML = "deploy/k8s/sim-job-gen.yaml"


def _kubectl(*args: str) -> str:
    return subprocess.run(
        ["kubectl", "--context", CONTEXT, *args], capture_output=True, text=True, check=True
    ).stdout


def _any_pod() -> str:
    for selector in ("--field-selector=status.phase=Running", ""):
        args = ["get", "pods", "-l", JOB_LABEL, "-o", "jsonpath={.items[0].metadata.name}"]
        if selector:
            args.insert(2, selector)
        out = _kubectl(*args).strip()
        if out:
            return out
    raise RuntimeError(
        f"{JOB_NAME} 파드를 하나도 못 찾음 — Job이 아직 생성 안 됐거나 전부 정리됨. "
        f"kubectl --context {CONTEXT} get pods -l {JOB_LABEL} -a 로 직접 확인."
    )


def _exec(pod: str, cmd: str) -> str:
    return _kubectl("exec", pod, "--", "sh", "-c", cmd)


def summary() -> tuple[set[str], set[str]]:
    recipes = json.load(open("data/recipes.json"))
    all_ids = {r["run_id"] for r in recipes}

    pod = _any_pod()
    done_out = _exec(
        pod,
        'for d in /data/raw/*/; do id=$(basename "$d"); '
        'if [ -f "$d/meta.json" ]; then echo "$id"; fi; done; true',
    )
    done = set(done_out.split()) & all_ids

    failed_out = _exec(pod, "cat /data/raw/failed_shard*.json 2>/dev/null || true")
    failed: set[str] = set()
    for line in failed_out.splitlines():
        line = line.strip()
        if line:
            try:
                failed.update(json.loads(line))
            except json.JSONDecodeError:
                pass

    pending = sorted(all_ids - done)
    pct = len(done) / len(all_ids) * 100

    print(
        f"전체 {len(all_ids)} / 완료 {len(done)} / 대기 {len(pending)} "
        f"/ 마지막 시도 실패 {len(failed)}"
    )
    print(f"진행률: {pct:.1f}%")
    if failed:
        print("최근 실패(자동 OnFailure 재시도로 이미 해소됐을 수 있음):", sorted(failed))
    if pending and len(pending) <= 10:
        print("대기 중:", pending)

    job_status = json.loads(_kubectl("get", "job", JOB_NAME, "-o", "json"))["status"]
    conditions = [c["type"] for c in job_status.get("conditions", [])]
    print(
        f"Job status: succeeded={job_status.get('succeeded', 0)} "
        f"failed={job_status.get('failed', 0)} active={job_status.get('active', 0)} "
        f"conditions={conditions}"
    )
    return done, failed


def retry() -> None:
    job_status = json.loads(_kubectl("get", "job", JOB_NAME, "-o", "json"))["status"]
    conditions = [c["type"] for c in job_status.get("conditions", [])]
    if "Failed" not in conditions:
        print("Job이 Failed 상태가 아님 — 재시도 불필요(OnFailure가 자동으로 처리 중일 수 있음).")
        return
    print(f"Job {JOB_NAME}이 Failed 상태 — delete 후 재적용(완료분은 meta.json 존재로 자동 skip).")
    _kubectl("delete", "job", JOB_NAME)
    subprocess.run(
        ["kubectl", "--context", CONTEXT, "apply", "-f", JOB_YAML], check=True
    )
    print("재적용 완료.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry", action="store_true")
    args = ap.parse_args()
    summary()
    if args.retry:
        retry()


if __name__ == "__main__":
    main()
