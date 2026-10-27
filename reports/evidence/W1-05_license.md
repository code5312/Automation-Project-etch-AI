# W1-05 증거: 라이선스

- 날짜: 2026-10-27
- `LICENSE`(Apache-2.0 전문), `NOTICE`(ViennaPS/ViennaLS GPL-3.0 예외 범위 설명 + 의존성 라이선스 표) 작성.
- 결정 배경: 사용자가 "GPL로 하는데 네가 보기에 Apache가 나으면 그걸로 해도 된다"고 위임. ViennaPS는 분리된 시뮬레이션 컨테이너에서 서브프로세스/바인딩 호출로만 쓰고 저장소 코드에 정적 링크·포함하지 않으므로, 저장소 코드 자체에는 Apache-2.0을 적용하고 `docker/Dockerfile.sim`(ViennaPS 포함) 이미지에만 GPL-3.0 조건이 적용됨을 NOTICE에 명시했다. 상세 근거는 `docs/project-definition.md` §11.
- 의존성 라이선스 표(`NOTICE`)는 각 패키지의 PyPI 공개 메타데이터 기준이며, 버전별 재확인은 하지 않았다 — [미확인]으로 NOTICE에 표기.
- 법률 자문을 거친 결정은 아니다 — NOTICE·definition.md 모두에 명시.
