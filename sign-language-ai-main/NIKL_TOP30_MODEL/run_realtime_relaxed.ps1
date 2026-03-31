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
