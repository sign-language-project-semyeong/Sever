# =========================
# 1. 프로젝트 폴더로 이동
# =========================
cd C:\Users\user\Desktop\ai30701\sign-language-ai


# =========================
# 2. 가상환경 생성
# =========================
python -m venv .venv


# =========================
# 3. 가상환경 활성화
# =========================
.\.venv\Scripts\Activate.ps1


# =========================
# 4. pip 업그레이드
# =========================
python -m pip install --upgrade pip


# =========================
# 5. PyTorch 설치 (CUDA 12.8)
# =========================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


# =========================
# 6. 나머지 필수 패키지 설치
# =========================
pip install mediapipe opencv-python pandas numpy tqdm scikit-learn matplotlib


# =========================
# 7. 데이터 폴더 구조 생성
# =========================
mkdir data
mkdir data\raw
mkdir data\raw\videos
mkdir data\raw\labels
mkdir data\interim
mkdir data\interim\manifests
mkdir data\interim\landmarks


# =========================
# 8. GPU 확인
# =========================
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"


# =========================
# 9. build_manifest.py 도움말 확인
# =========================
python src/preprocess/build_manifest.py --help


# =========================
# 10. extract_landmarks.py 도움말 확인
# =========================
python src/preprocess/extract_landmarks.py --help


# =========================
# 11. filter_topk_labels.py 도움말 확인
# =========================
python src/preprocess/filter_topk_labels.py --help


# =========================
# 12. train.py 도움말 확인
# =========================
python -m src.train.train --help


# 한번에 하는거

cd C:\Users\user\Desktop\ai30701\sign-language-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install mediapipe opencv-python pandas numpy tqdm scikit-learn matplotlib
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
python src/preprocess/build_manifest.py --help
python src/preprocess/extract_landmarks.py --help
python src/preprocess/filter_topk_labels.py --help
python -m src.train.train --help



# 여기서부터 내가 만들었다 은찬아



# =========================
# 13. Current Workflow
# =========================

This repo currently has two tracks.

1. Word/gloss classification pipeline
- `manifest.csv`, `manifest_top10.csv`, `manifest_top50.csv`
- Gesture-level samples
- Used as the main baseline for realtime subtitle-style inference

2. Sentence-level experimental pipeline
- `sentence_manifest.jsonl`
- Full sentence sequence experiments
- Not the current main product direction


# =========================
# 14. Data Layout
# =========================

The raw NIKL dataset may contain `json` and `mp4` files mixed together.
In this project they are separated like this:

```text
data/
  raw/
    videos/
    labels/
  interim/
    manifests/
    landmarks_top10/
    landmarks_top50/
    sentence_landmarks/
```

Example raw source path:

```text
G:\NIKL_Sign Language Parallel Corpus_2024_FI_MB\NIKL_Sign Language Parallel Corpus_2024_FI_MB
```

Project usage:
- `mp4` files -> `data/raw/videos`
- `json` files -> `data/raw/labels`


# =========================
# 15. Manifest Types
# =========================

1. `data/interim/manifests/manifest.csv`
- Full gesture-level manifest
- 33325 samples
- 942 labels

2. `data/interim/manifests/manifest_top10.csv`
- Top 10 labels only
- Fast validation/debugging

3. `data/interim/manifests/manifest_top50.csv`
- Top 50 labels only
- Main dataset for realtime subtitle-style model

4. `data/interim/manifests/sentence_manifest.jsonl`
- Sentence-level manifest
- Stores full sentence range and gloss sequence
- Used for CTC experiments


# =========================
# 16. Current Product Direction
# =========================

Current target flow:

1. Recognize signs at word/gloss level
2. Accumulate recognized words in realtime
3. Show them like subtitles
4. Close the sentence after a short pause
5. Send the final word sequence to Gemini API for natural sentence polishing

So the current main direction is:
`word classifier + subtitle accumulation + Gemini post-processing`


# =========================
# 17. Top50 Data Prep
# =========================

Create Top50 manifest:

```powershell
python src/preprocess/filter_topk_labels.py `
  --input_csv data\interim\manifests\manifest.csv `
  --output_csv data\interim\manifests\manifest_top50.csv `
  --label_column label `
  --top_k 50 `
  --min_samples 20
```

Extract Top50 landmarks:

PowerShell
```powershell
python src/preprocess/extract_landmarks.py `
  --manifest_csv data\interim\manifests\manifest_top50.csv `
  --save_root data\interim\landmarks_top50 `
  --max_frames 30 `
  --model_asset_path models\mediapipe\hand_landmarker.task `
  --num_workers 3
```

bash
```bash
./.venv/Scripts/python.exe src/preprocess/extract_landmarks.py \
  --manifest_csv data/interim/manifests/manifest_top50.csv \
  --save_root data/interim/landmarks_top50 \
  --max_frames 30 \
  --model_asset_path models/mediapipe/hand_landmarker.task \
  --num_workers 3
```

Notes:
- `extract_landmarks.py` now supports multiprocessing
- Existing `.npy` files are skipped automatically
- You can rerun the same command to resume work


# =========================
# 18. Top50 Training
# =========================

If GPU is available, training should print:

```text
device: cuda
```

Top50 training command:

PowerShell
```powershell
python -m src.train.train `
  --manifest_csv data\interim\manifests\manifest_top50.csv `
  --landmark_root data\interim\landmarks_top50 `
  --label_column label `
  --max_len 30 `
  --epochs 5 `
  --batch_size 6 `
  --lr 0.001 `
  --checkpoint_dir models\checkpoints_top50 `
  --num_workers 4
```

bash
```bash
./.venv/Scripts/python.exe -m src.train.train \
  --manifest_csv data/interim/manifests/manifest_top50.csv \
  --landmark_root data/interim/landmarks_top50 \
  --label_column label \
  --max_len 30 \
  --epochs 5 \
  --batch_size 6 \
  --lr 0.001 \
  --checkpoint_dir models/checkpoints_top50 \
  --num_workers 4
```

Current result example:
- usable samples: 21368
- num classes: 50
- best val_acc: 0.6203

Checkpoint:

```text
models/checkpoints_top50/best_gru_model.pt
```


# =========================
# 19. Realtime Subtitle Inference
# =========================

`src/infer/realtime_infer.py` currently includes:
- top-k prediction display
- recent prediction voting
- commit only after a label stays stable for several frames
- duplicate word suppression
- sentence close after 3 seconds of no input
- subtitle buffer display

Run:

```bash
./.venv/Scripts/python.exe -m src.infer.realtime_infer \
  --checkpoint models/checkpoints_top50/best_gru_model.pt \
  --model_asset_path models/mediapipe/hand_landmarker.task \
  --mirror \
  --threshold 0.7 \
  --stable_frames 8 \
  --vote_window 12 \
  --cooldown_frames 12 \
  --sentence_timeout 3 \
  --min_token_gap 0.8
```

Keys:
- `q`: quit
- `c`: clear subtitle buffer


# =========================
# 20. Sentence-Level Experimental Pipeline
# =========================

Additional sentence-level files:
- `src/preprocess/build_sentence_manifest.py`
- `src/preprocess/extract_sentence_landmarks.py`
- `src/datasets/sentence_dataset.py`
- `src/models/ctc_model.py`
- `src/train/train_ctc.py`
- `src/infer/realtime_sequence_infer.py`

This is for CTC-based sequence experiments.
It is not the current main service path.

Current main service path:
- Top50 word classifier
- Realtime subtitle accumulation
- Gemini API post-processing


# =========================
# 21. Current Outputs
# =========================

Key outputs:

```text
data/interim/manifests/manifest.csv
data/interim/manifests/manifest_top10.csv
data/interim/manifests/manifest_top50.csv
data/interim/manifests/sentence_manifest.jsonl
data/interim/landmarks_top10
data/interim/landmarks_top50
data/interim/sentence_landmarks
models/mediapipe/hand_landmarker.task
models/checkpoints_top10/best_gru_model.pt
models/checkpoints_top50/best_gru_model.pt
```


# =========================
# 22. Next Steps
# =========================

Recommended next steps:
1. Stabilize Top50 realtime subtitle inference
2. Connect Gemini API
3. Post-process confirmed word sequences into natural sentences
4. Expand to Top100 later if needed
