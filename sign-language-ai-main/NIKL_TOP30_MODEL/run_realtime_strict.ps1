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

