# NIKL Top30 Modelset

이 폴더는 현재 프로젝트 기준으로 전달 가능한 최소 모델셋입니다.

포함 파일:
- `models/best_gru_model.pt`: 학습 완료된 GRU 체크포인트
- `models/hand_landmarker.task`: MediaPipe 손 랜드마커 모델
- `configs/nikl.yaml`: 데이터셋 기본 경로 설정
- `configs/nikl_gru.yaml`: 학습 설정값
- `meta/manifest_top30.csv`: top30 학습에 사용한 manifest
- `meta/labels_top30.txt`: top30 라벨 목록과 샘플 수

현재 모델 정보:
- 데이터셋: NIKL 원본 데이터 기반 top30
- 입력: 양손 3D 랜드마크 126차원
- 모델: GRU
- max_len: 30
- 클래스 수: 30
- 검증 정확도: `val_acc = 0.6341`

실시간 테스트 명령:

```powershell
python -m src.infer.realtime_infer `
  --checkpoint release\nikl_top30_modelset\models\best_gru_model.pt `
  --model_asset_path release\nikl_top30_modelset\models\hand_landmarker.task `
  --top_k 5 `
  --threshold 0.80 `
  --min_margin 0.20 `
  --stable_frames 8 `
  --vote_window 10 `
  --cooldown_frames 8 `
  --mirror
```

완화 버전 테스트 명령:

```powershell
python -m src.infer.realtime_infer `
  --checkpoint release\nikl_top30_modelset\models\best_gru_model.pt `
  --model_asset_path release\nikl_top30_modelset\models\hand_landmarker.task `
  --top_k 5 `
  --threshold 0.70 `
  --min_margin 0.10 `
  --stable_frames 6 `
  --vote_window 8 `
  --cooldown_frames 6 `
  --mirror
```

주의:
- 이 모델은 완성형 문장 번역기가 아니라 단어 후보 생성 모델입니다.
- 실시간에서는 조명, 거리, 손 위치에 따라 흔들릴 수 있습니다.
- 현재 기준으로는 후보 단어 생성용 프로토타입으로 보는 것이 맞습니다.
