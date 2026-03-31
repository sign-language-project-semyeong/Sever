# ab_final20

실시간 수어 입력에서 제한된 어휘의 단어 후보를 생성하는 프로토타입 모델입니다.

## 구성 파일

- `best_gru_model.pt`: 최종 학습 체크포인트
- `hand_landmarker.task`: MediaPipe 손 랜드마커 모델
- `ab_final20_labels.yaml`: 최종 20개 라벨 목록
- `ab_final20_gru.yaml`: 학습 설정 참고 파일
- `realtime_infer.py`: 실시간 추론 스크립트
- `requirements.txt`: Python 의존성

## 모델 성격

- 완성형 수어 번역기가 아니라, 제한된 단어셋 안에서 단어 후보를 생성하는 실험/프로토타입 모델입니다.
- 서버 후처리 또는 문장화 모델(Gemini 등) 이전 단계의 전단 모델로 사용하는 것을 전제로 합니다.

## 최종 단어셋

- `글1`
- `쓰다1`
- `읽다1`
- `필요1`
- `가족2`
- `친구1`
- `모으다1`
- `말하다1`
- `놀다1`
- `집1`
- `먹다1`
- `오늘1`
- `버스1`
- `여행1`
- `가다1`
- `공부1`
- `학교1`
- `책1`
- `일하다1`
- `지하철1`

## 실행 예시

```powershell
python -m src.infer.realtime_infer `
  --checkpoint models\checkpoints_ab_final20\best_gru_model.pt `
  --model_asset_path models\mediapipe\hand_landmarker.task `
  --top_k 5 `
  --threshold 0.80 `
  --min_margin 0.25 `
  --stable_frames 12 `
  --vote_window 16 `
  --cooldown_frames 16 `
  --mirror
```

## 참고

- 현재 버전은 동작에 따라 후보 단어가 바뀌는지 확인하는 데모용 기준선입니다.
- 특정 단어가 과하게 자주 뜨는 편향이나, 비슷한 동작 사이 후보 흔들림은 일부 남아 있을 수 있습니다.
