"""
새 수어 단어 녹화 + 재학습 스크립트
======================================
실행: py record_new_words.py

조작법:
  SPACE  : 샘플 녹화 시작
  Q      : 현재 단어 건너뛰기 / 종료
  ESC    : 즉시 종료
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.preprocess.extract_landmarks import HandLandmarkExtractor

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
LANDMARK_DIR  = BASE_DIR / "demo_gesture_2026-03-31_v1" / "landmarks"
MANIFEST_PATH = BASE_DIR / "demo_gesture_2026-03-31_v1" / "manifests" / "demo_manifest_clean.csv"
HAND_TASK     = BASE_DIR / "ab_final20" / "hand_landmarker.task"

# ── 녹화 설정 ──────────────────────────────────────────────────────────────────
SAMPLES_PER_WORD = 30   # 단어당 샘플 수
RECORD_FRAMES    = 45   # 샘플 당 프레임 수 (max_len 과 맞춤)
COUNTDOWN_SEC    = 2    # 녹화 전 카운트다운 (초)
FPS_TARGET       = 30.0

# ── 추가할 단어 목록 ───────────────────────────────────────────────────────────
# 여기에 녹화할 단어를 추가해라!
NEW_WORDS = [
    "네",
    "아니요",
    "좋아요",
    "싫어요",
    "이름",
    "도움",
    "천천히",
    "다시",
    "괜찮아요",
    "모르다",
]


def get_existing_sample_count(word: str) -> int:
    """기존 manifest에서 해당 단어의 샘플 수 조회"""
    if not MANIFEST_PATH.exists():
        return 0
    count = 0
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label") == word:
                count += 1
    return count


def append_to_manifest(sample_id: str, label: str, num_frames: int) -> None:
    write_header = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "label", "source", "camera_index", "num_frames", "captured_at"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "sample_id":    sample_id,
            "label":        label,
            "source":       "webcam_demo",
            "camera_index": 0,
            "num_frames":   num_frames,
            "captured_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })


def draw_ui(frame, word: str, sample_idx: int, total: int, status: str, score: float = 0.0):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 상단 바
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"단어: {word}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 100), 2)
    cv2.putText(frame, f"샘플: {sample_idx}/{total}   {status}", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

    # 하단 안내
    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "SPACE: 녹화  |  Q: 건너뛰기  |  ESC: 종료", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


def record_sample(cap, extractor: HandLandmarkExtractor, word: str, sample_idx: int) -> np.ndarray | None:
    """카운트다운 후 RECORD_FRAMES 프레임 녹화 → (T, 126) 랜드마크 배열 반환"""
    # 카운트다운
    start = time.time()
    while time.time() - start < COUNTDOWN_SEC:
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        remaining = COUNTDOWN_SEC - int(time.time() - start)
        h, w = frame.shape[:2]
        draw_ui(frame, word, sample_idx, SAMPLES_PER_WORD, f"준비... {remaining}")
        cv2.putText(frame, str(remaining), (w // 2 - 30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 100, 255), 6)
        cv2.imshow("수어 녹화기", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return None

    # 녹화
    frames: list[np.ndarray] = []
    for i in range(RECORD_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        feature = extractor.extract_from_frame(frame, fps=FPS_TARGET)[0]
        frames.append(feature)

        progress = int((i + 1) / RECORD_FRAMES * (frame.shape[1] - 20))
        draw_ui(frame, word, sample_idx, SAMPLES_PER_WORD, f"녹화 중 {i+1}/{RECORD_FRAMES}")
        cv2.rectangle(frame, (10, frame.shape[0] - 55), (10 + progress, frame.shape[0] - 45),
                      (0, 255, 100), -1)
        cv2.imshow("수어 녹화기", frame)
        cv2.waitKey(1)

    return np.array(frames, dtype=np.float32) if frames else None


def record_word(cap, extractor: HandLandmarkExtractor, word: str) -> int:
    """단어 하나를 SAMPLES_PER_WORD 개 녹화. 완료된 샘플 수 반환."""
    existing = get_existing_sample_count(word)
    print(f"\n[녹화] '{word}'  기존: {existing}샘플")

    recorded = 0
    sample_start_idx = existing

    while recorded < SAMPLES_PER_WORD:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        draw_ui(frame, word, sample_start_idx + recorded, SAMPLES_PER_WORD,
                "SPACE 눌러서 녹화 시작")
        cv2.imshow("수어 녹화기", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            print(f"  → '{word}' 건너뜀 ({recorded}개 완료)")
            return recorded
        if key == ord(" "):
            sample_idx = sample_start_idx + recorded
            landmarks = record_sample(cap, extractor, word, sample_idx + 1)
            if landmarks is None:
                continue

            sample_id = f"demo_{word}_{sample_idx:04d}"
            save_path = LANDMARK_DIR / f"{sample_id}.npy"
            np.save(save_path, landmarks)
            append_to_manifest(sample_id, word, len(landmarks))
            recorded += 1
            print(f"  ✅ {sample_id} 저장 ({recorded}/{SAMPLES_PER_WORD})")

    print(f"  → '{word}' 완료! {recorded}개 녹화됨")
    return recorded


def run_training():
    """재학습 실행"""
    import subprocess
    checkpoint_dir = BASE_DIR / "demo_gesture_2026-03-31_v1" / "models"
    print("\n재학습 시작...")

    result = subprocess.run(
        [sys.executable, "-m", "src.train.train",
         "--manifest_csv",   str(MANIFEST_PATH),
         "--landmark_root",  str(LANDMARK_DIR),
         "--checkpoint_dir", str(checkpoint_dir),
         "--max_len",        "45",
         "--epochs",         "40",
         "--batch_size",     "8",
         "--hidden_size",    "128",
         "--num_layers",     "2",
         "--dropout",        "0.2",
         "--lr",             "0.001",
         "--label_smoothing","0.02",
         "--val_ratio",      "0.2",
         "--seed",           "42",
        ],
        cwd=str(BASE_DIR),
        capture_output=False,
    )
    if result.returncode == 0:
        print("재학습 완료! ai_server.py 재시작하면 새 단어가 적용돼.")
    else:
        print("재학습 중 오류 발생.")


def main():
    print("=" * 50)
    print("  수어 단어 녹화기")
    print("=" * 50)
    print(f"녹화할 단어 목록: {NEW_WORDS}")
    print(f"단어당 샘플: {SAMPLES_PER_WORD}개")
    print()

    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

    extractor = HandLandmarkExtractor(max_num_hands=2, model_asset_path=str(HAND_TASK))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    total_recorded = 0
    try:
        for word in NEW_WORDS:
            n = record_word(cap, extractor, word)
            total_recorded += n

            # 다음 단어로 넘어가기 전 잠깐 대기
            if n > 0:
                time.sleep(1.0)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()

    print(f"\n총 {total_recorded}개 샘플 녹화 완료.")

    if total_recorded > 0:
        ans = input("\n재학습 바로 진행할까요? (y/n): ").strip().lower()
        if ans == "y":
            run_training()
        else:
            print("\n나중에 재학습하려면 아래 명령어 실행:")
            print(f"  py -m src.train.train --config demo_gesture_2026-03-31_v1/configs/demo_gesture_gru.yaml")


if __name__ == "__main__":
    main()
