from __future__ import annotations

import argparse
import re
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.models.gru_model import GRUSignClassifier
from src.preprocess.extract_landmarks import HandLandmarkExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints_top50/best_gru_model.pt"))
    parser.add_argument("--model_asset_path", type=Path, default=Path("models/mediapipe/hand_landmarker.task"))
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min_margin", type=float, default=0.20)
    parser.add_argument("--stable_frames", type=int, default=6)
    parser.add_argument("--vote_window", type=int, default=8)
    parser.add_argument("--cooldown_frames", type=int, default=8)
    parser.add_argument("--sentence_timeout", type=float, default=3.0)
    parser.add_argument("--min_token_gap", type=float, default=0.8)
    parser.add_argument("--idle_frames", type=int, default=8)
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GRUSignClassifier, dict[int, str], int, int, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GRUSignClassifier(
        input_size=int(checkpoint.get("input_size", 126)),
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        num_layers=int(checkpoint.get("num_layers", 2)),
        num_classes=int(checkpoint["num_classes"]),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx2label_raw = checkpoint["idx2label"]
    idx2label = {int(k): str(v) for k, v in idx2label_raw.items()}
    max_len = int(checkpoint.get("max_len", 30))
    input_size = int(checkpoint.get("input_size", 126))
    normalize_landmarks = bool(checkpoint.get("normalize_landmarks", True))
    return model, idx2label, max_len, input_size, normalize_landmarks


def normalize_hand_block(hand_block: np.ndarray) -> np.ndarray:
    if hand_block.shape != (21, 3):
        raise ValueError(f"Expected hand block shape=(21, 3), got {hand_block.shape}")

    if np.abs(hand_block).sum() < 1e-6:
        return hand_block

    wrist = hand_block[0].copy()
    hand_block = hand_block - wrist

    finger_points = hand_block[1:]
    norms = np.linalg.norm(finger_points, axis=1)
    valid_norms = norms[norms > 1e-6]
    scale = float(valid_norms.max()) if len(valid_norms) > 0 else 1.0
    if scale < 1e-6:
        scale = 1.0

    return hand_block / scale


def normalize_sequence(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] != 126:
        return arr

    frames = arr.reshape(arr.shape[0], 2, 21, 3).copy()
    for t in range(frames.shape[0]):
        for hand_idx in range(frames.shape[1]):
            frames[t, hand_idx] = normalize_hand_block(frames[t, hand_idx])

    return frames.reshape(arr.shape[0], -1).astype(np.float32)


def pad_sequence(sequence: list[np.ndarray], max_len: int, input_size: int) -> np.ndarray:
    if not sequence:
        return np.zeros((1, max_len, input_size), dtype=np.float32)

    arr = np.vstack(sequence).astype(np.float32)
    if len(arr) > max_len:
        arr = arr[-max_len:]
    elif len(arr) < max_len:
        pad = np.zeros((max_len - len(arr), input_size), dtype=np.float32)
        arr = np.vstack([pad, arr])

    return np.expand_dims(arr, axis=0)


def predict_topk(model, sequence_tensor: torch.Tensor, idx2label: dict[int, str], top_k: int) -> list[tuple[str, float]]:
    with torch.no_grad():
        logits = model(sequence_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(top_k, probs.shape[0])
        values, indices = torch.topk(probs, k=k)

    return [
        (idx2label[int(index)], float(value))
        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
    ]


def is_confident_prediction(predictions: list[tuple[str, float]], threshold: float, min_margin: float) -> tuple[bool, float]:
    if not predictions:
        return False, 0.0

    top_score = predictions[0][1]
    second_score = predictions[1][1] if len(predictions) > 1 else 0.0
    margin = top_score - second_score
    return top_score >= threshold and margin >= min_margin, margin


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def prettify_label(label: str) -> str:
    cleaned = re.sub(r"\d+$", "", label)
    cleaned = cleaned.replace("#", "").strip()
    return cleaned or label


def smooth_candidate(history: deque[str], min_votes: int) -> str:
    if not history:
        return ""
    label, votes = Counter(history).most_common(1)[0]
    return label if votes >= min_votes else ""


def draw_predictions(
    frame: np.ndarray,
    predictions: list[tuple[str, float]],
    candidate_label: str,
    current_tokens: list[str],
    completed_sentences: deque[str],
    live_status: str,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    title_font = load_font(30)
    body_font = load_font(24)
    subtitle_font = load_font(28)

    header = f"현재 단어: {candidate_label if candidate_label else '-'}"
    draw.text((20, 20), header, font=title_font, fill=(0, 255, 0))

    status_text = f"상태: {live_status}"
    draw.text((20, 56), status_text, font=body_font, fill=(180, 220, 255))

    pred_y = 92
    if predictions:
        for idx, (label, score) in enumerate(predictions, start=1):
            color = (255, 255, 0) if idx == 1 else (210, 210, 210)
            line = f"{idx}. {prettify_label(label)} ({score:.2f})"
            draw.text((20, pred_y), line, font=body_font, fill=color)
            pred_y += 30
    else:
        draw.text((20, pred_y), "신뢰도 높은 예측 없음", font=body_font, fill=(160, 160, 160))

    subtitle_y = frame.shape[0] - 180
    current_sentence = " ".join(current_tokens) if current_tokens else ""
    current_text = f"실시간 자막: {current_sentence}" if current_sentence else "실시간 자막: "
    draw.text(
        (20, subtitle_y),
        current_text,
        font=subtitle_font,
        fill=(255, 255, 255) if current_sentence else (180, 180, 180),
    )

    history_y = subtitle_y + 40
    completed_list = list(completed_sentences)
    if completed_list:
        draw.text((20, history_y), "완성된 자막:", font=body_font, fill=(255, 220, 150))
        history_y += 30
        for sentence in completed_list[-4:]:
            draw.text((20, history_y), sentence, font=body_font, fill=(180, 220, 255))
            history_y += 30

    guide = "q: 종료 | c: 자막 버퍼 초기화"
    draw.text((20, frame.shape[0] - 35), guide, font=body_font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx2label, max_len, input_size, normalize_landmarks = load_model(args.checkpoint, device)
    window_size = args.window_size or max_len

    extractor = HandLandmarkExtractor(max_num_hands=2, model_asset_path=str(args.model_asset_path))
    frame_buffer: deque[np.ndarray] = deque(maxlen=window_size)
    recent_labels: deque[str] = deque(maxlen=args.vote_window)
    completed_sentences: deque[str] = deque(maxlen=20)
    current_tokens: list[str] = []

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        extractor.close()
        raise ValueError(f"Failed to open camera index: {args.camera_index}")

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"window_size: {window_size}")
    print(f"threshold: {args.threshold}")
    print(f"min_margin: {args.min_margin}")
    print("q: 종료, c: 자막 버퍼 초기화")

    candidate_label = ""
    stable_count = 0
    cooldown_count = 0
    no_hand_frames = 0
    last_token_time = time.time()
    last_commit_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 30.0

            frame_feature = extractor.extract_from_frame(frame, fps=fps)[0]
            has_hands = bool(np.any(np.abs(frame_feature) > 1e-6))
            frame_buffer.append(frame_feature)
            no_hand_frames = 0 if has_hands else no_hand_frames + 1

            sequence = pad_sequence(list(frame_buffer), max_len=max_len, input_size=input_size)
            if normalize_landmarks:
                sequence[0] = normalize_sequence(sequence[0])
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32, device=device)
            predictions = predict_topk(model, sequence_tensor, idx2label, args.top_k)

            top_label = ""
            top_score = 0.0
            margin = 0.0
            if predictions:
                top_label, top_score = predictions[0]
            confident_prediction, margin = is_confident_prediction(predictions, args.threshold, args.min_margin)

            display_predictions: list[tuple[str, float]] = []

            if has_hands and len(frame_buffer) >= max_len and confident_prediction:
                recent_labels.append(top_label)
            elif recent_labels:
                recent_labels.clear()

            smoothed_label = smooth_candidate(recent_labels, max(1, args.vote_window // 2))
            display_candidate = prettify_label(smoothed_label) if smoothed_label else ""
            live_status = "대기 중"

            if smoothed_label and cooldown_count == 0:
                if smoothed_label == candidate_label:
                    stable_count += 1
                else:
                    candidate_label = smoothed_label
                    stable_count = 1

                live_status = f"단어 추적 중: {prettify_label(candidate_label)} ({stable_count}/{args.stable_frames})"
                if stable_count >= args.stable_frames:
                    now = time.time()
                    allow_commit = (now - last_commit_time) >= args.min_token_gap
                    display_token = prettify_label(candidate_label)
                    if allow_commit and (not current_tokens or current_tokens[-1] != display_token):
                        current_tokens.append(display_token)
                        last_token_time = now
                        last_commit_time = now
                        live_status = f"자막에 추가됨: {display_token}"
                    else:
                        live_status = f"중복 방지 대기: {display_token}"
                    cooldown_count = args.cooldown_frames
                    stable_count = 0
                    recent_labels.clear()
            else:
                if not has_hands:
                    candidate_label = ""
                    live_status = "손이 감지되지 않음"
                elif top_score < args.threshold:
                    live_status = f"신뢰도 낮음 ({top_score:.2f})"
                elif margin < args.min_margin:
                    live_status = f"후보 경합 중 (차이 {margin:.2f})"
                elif len(frame_buffer) < max_len:
                    live_status = f"초기 프레임 수집 중 ({len(frame_buffer)}/{max_len})"
                stable_count = 0

            if has_hands and len(frame_buffer) >= max_len:
                display_predictions = predictions

            if cooldown_count > 0:
                cooldown_count -= 1

            now = time.time()
            if current_tokens and no_hand_frames >= args.idle_frames and (now - last_token_time) >= args.sentence_timeout:
                completed_sentences.append(" ".join(current_tokens))
                current_tokens = []
                candidate_label = ""
                stable_count = 0
                cooldown_count = 0
                no_hand_frames = 0
                recent_labels.clear()
                frame_buffer.clear()
                live_status = "문장 확정됨"

            display = draw_predictions(
                frame=frame,
                predictions=display_predictions,
                candidate_label=(
                    display_candidate
                    if display_candidate
                    else (
                        prettify_label(top_label)
                        if has_hands and len(frame_buffer) >= max_len and top_label and confident_prediction
                        else ""
                    )
                ),
                current_tokens=current_tokens,
                completed_sentences=completed_sentences,
                live_status=live_status,
            )
            cv2.imshow("수어 실시간 자막", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                frame_buffer.clear()
                recent_labels.clear()
                current_tokens = []
                completed_sentences.clear()
                candidate_label = ""
                stable_count = 0
                cooldown_count = 0
                last_token_time = time.time()
                last_commit_time = 0.0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    main()
