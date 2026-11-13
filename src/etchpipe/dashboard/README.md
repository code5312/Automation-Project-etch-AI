# EtchGuard 대시보드

서빙 환경을 실행하고 브라우저에서 FastAPI의 루트 주소를 연다. 기본 Compose 주소는 `http://localhost:8000/`이다. 서버 실행과 등록 모델 준비는 [서빙 가이드](../../../docs/serving.md)를 참고한다.

- **레시피 예측**은 같은 출처의 `POST /api/predict`를 호출한다. 서버는 현재 모델의 시간별 윤곽과 마지막 시점의 지표를 반환한다. Golden10 `r0007`의 입력값과 일치하면 2분·5분 ViennaPS 기준 윤곽을 같은 축척으로 겹쳐 볼 수 있다. PNG에 표시 시점, 모델·기준 출처, 좌표 단위, 제출한 레시피 값과 단위가 기록된다.
- **모델 비교**는 `GET /api/models/compare`로 W6-12 결과를 읽는다. 서빙 프로세스에서 `reports/eval/W6-12_model_comparison.json`에 접근할 수 있도록 파일을 준비한다. 화면에 평가 체크포인트, 분할, 지표 정의가 표시된다.
- **게이트 이력**은 `GET /api/gate/history`로 서빙 환경의 `registry/history.jsonl`을 읽는다. **이력 새로고침**으로 다시 조회할 수 있다.

팀 환경에서는 `/health`를 확인한 뒤 인증된 `/model`과 브라우저 예측 응답의 모델 ID를 대조한다. `r0007`의 2분·5분 응답, 동일 축척 겹치기, PNG, 모델 비교 및 게이트 이력을 확인한다. 재학습 모델을 서빙할 때는 카드와 일치하는 체크포인트를 `runs/retrain/<model_id>/model.pt`에 연결하고, 서빙을 재시작한 뒤 모델 ID를 다시 확인한다.
