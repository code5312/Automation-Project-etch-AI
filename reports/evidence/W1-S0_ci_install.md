# W1-S0: CI에서 ViennaPS·PyG·torch_scatter 설치 성공

- 목표: GitHub Actions(ubuntu-latest, 호스트와 무관한 실제 CI 환경)에서 `pr-check`의 4개 잡이 모두 통과하는지 확인. 특히 `viennaps-smoke`(Python 3.13 + ViennaPS/ViennaLS + apt 패키지)와 `unit-tests`(torch_scatter 등 무거운 빌드는 없음, 가벼운 환경)가 로컬 컨테이너 드라이런(W1-04)과 동일하게 재현되는지가 핵심.
- 시간 상자: 1시간
- 실행 명령: 없음(이미 push됨) — 확인만 수행
  ```
  curl https://api.github.com/repos/code5312/Automation-Project-etch-AI/actions/runs?per_page=10
  curl https://api.github.com/repos/code5312/Automation-Project-etch-AI/actions/runs/40468419349/jobs
  ```
- 측정값(실제 GitHub Actions 실행 결과, run id 40468419349, 2026-10-27T07:39Z, branch main):

| 잡 | 결과 | 소요 |
|---|---|---|
| lint | success | 7s |
| forbidden-expressions | success | 5s |
| unit-tests | success | 11s |
| viennaps-smoke | success | 21s |

- 판정: **통과(PASS)**. `pr-check` 전체(`conclusion: success`)이고 4개 잡 모두 개별적으로 success. 로컬 컨테이너 테스트(W1-03/W1-04 증거)와 실제 CI 결과가 일치함을 확인 — "로컬 Docker와 GitHub Actions 러너 환경 차이" 우려(W1-04 증거의 미확인 항목)가 해소됨.
- 대안: 불필요(통과).
- torch_scatter 소스 빌드(=train 환경, W1-03)는 이번 pr-check에 포함되지 않는다(pr-check는 sim 환경만 스모크 테스트). train 환경의 CI 재현은 이번 체크리스트 항목(S0)의 범위 밖(CLAUDE.md는 S0을 "ViennaPS·PyG·torch_scatter 설치 성공"으로 묶어 서술하지만, 실제 pr-check 워크플로는 sim 스모크만 포함하도록 설계됨 — DOC 공백, 보고에 기록).
