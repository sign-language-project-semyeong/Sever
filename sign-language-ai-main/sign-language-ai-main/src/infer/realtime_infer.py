from __future__ import annotations

import argparse
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
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--stable_frames", type=int, default=8)
    parser.add_argument("--vote_window", type=int, default=12)
    parser.add_argument("--cooldown_frames", type=int, default=12)
    parser.add_argument("--sentence_timeout", type=float, default=3.0)
    parser.add_argument("--min_token_gap", type=float, default=0.8)
    parser.add_argument("--show_debug", action="store_true")
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GRUSignClassifier, dict[int, str], int, int]:
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
    return model, idx2label, max_len, input_size


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


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def smooth_candidate(history: deque[str], min_votes: int) -> str:
    if not history:
        return ""
    label, votes = Counter(history).most_common(1)[0]
    return label if votes >= min_votes else ""


def draw_predictions(
    frame: np.ndarray,
    predictions: list[tuple[str, float]],
    candidate_label: str,
    candidate_score: float,
    has_hands: bool,
    current_tokens: list[str],
    completed_sentences: deque[str],
    show_debug: bool,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    title_font = load_font(32)
    body_font = load_font(24)
    subtitle_font = load_font(34)
    status_font = load_font(22)

    top_panel_h = 110
    subtitle_panel_h = 150

    draw.rounded_rectangle((12, 12, frame.shape[1] - 12, top_panel_h), radius=18, fill=(10, 10, 10, 220))
    draw.rounded_rectangle(
        (12, frame.shape[0] - subtitle_panel_h - 12, frame.shape[1] - 12, frame.shape[0] - 12),
        radius=18,
        fill=(10, 10, 10, 220),
    )

    status_text = "HAND DETECTED" if has_hands else "NO HAND"
    status_color = (80, 220, 120) if has_hands else (220, 120, 120)
    draw.text((28, 24), status_text, font=status_font, fill=status_color)

    candidate_text = candidate_label if candidate_label else "..."
    draw.text((28, 52), f"Current: {candidate_text}", font=title_font, fill=(255, 255, 255))

    bar_x = 28
    bar_y = 88
    bar_w = min(320, frame.shape[1] - 56)
    bar_h = 12
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=6, fill=(55, 55, 55))
    fill_w = int(bar_w * max(0.0, min(1.0, candidate_score)))
    if fill_w > 0:
        draw.rounded_rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), radius=6, fill=(80, 220, 120))
    draw.text((bar_x + bar_w + 14, bar_y - 8), f"{candidate_score:.2f}", font=status_font, fill=(210, 210, 210))

    subtitle_y = frame.shape[0] - subtitle_panel_h
    current_sentence = " ".join(current_tokens) if current_tokens else ""
    current_text = current_sentence if current_sentence else "Waiting for committed words..."
    draw.text((28, subtitle_y + 18), "Subtitle", font=status_font, fill=(130, 200, 255))
    draw.text(
        (28, subtitle_y + 50),
        current_text,
        font=subtitle_font,
        fill=(255, 255, 255) if current_sentence else (160, 160, 160),
    )

    history_y = subtitle_y + 100
    for sentence in list(completed_sentences)[-2:]:
        draw.text((28, history_y), sentence, font=body_font, fill=(180, 220, 255))
        history_y += 30

    if show_debug and predictions:
        debug_lines = []
        for label, score in predictions[:3]:
            debug_lines.append(f"{label} {score:.2f}")
        debug_text = " | ".join(debug_lines)
        draw.text((28, top_panel_h + 12), debug_text, font=status_font, fill=(210, 210, 80))

    guide = "q quit | c clear"
    draw.text((frame.shape[1] - 190, frame.shape[0] - 42), guide, font=status_font, fill=(200, 200, 200))

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx2label, max_len, input_size = load_model(args.checkpoint, device)
    window_size = args.window_size or max_len

    extractor = HandLandmarkExtractor(max_num_hands=2, model_asset_path=str(args.model_asset_path))
    frame_buffer: deque[np.ndarray] = deque(maxlen=window_size)
    recent_labels: deque[str] = deque(maxlen=args.vote_window)
    completed_sentences: deque[str] = deque(maxlen=3)
    current_tokens: list[str] = []

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        extractor.close()
        raise ValueError(f"Failed to open camera index: {args.camera_index}")

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"window_size: {window_size}")
    print("Press q to quit, c to clear the subtitle buffer.")

    candidate_label = ""
    stable_count = 0
    cooldown_count = 0
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

            sequence = pad_sequence(list(frame_buffer), max_len=max_len, input_size=input_size)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32, device=device)
            predictions = predict_topk(model, sequence_tensor, idx2label, args.top_k)

            top_label = ""
            top_score = 0.0
            if predictions:
                top_label, top_score = predictions[0]

            if has_hands and len(frame_buffer) >= max_len and top_score >= args.threshold:
                recent_labels.append(top_label)
            elif recent_labels:
                recent_labels.clear()

            smoothed_label = smooth_candidate(recent_labels, max(1, args.vote_window // 2))

            if smoothed_label and cooldown_count == 0:
                if smoothed_label == candidate_label:
                    stable_count += 1
                else:
                    candidate_label = smoothed_label
                    stable_count = 1

                if stable_count >= args.stable_frames:
                    now = time.time()
                    allow_commit = (now - last_commit_time) >= args.min_token_gap
                    if allow_commit and (not current_tokens or current_tokens[-1] != candidate_label):
                        current_tokens.append(candidate_label)
                        last_token_time = now
                        last_commit_time = now
                    cooldown_count = args.cooldown_frames
                    stable_count = 0
                    recent_labels.clear()
            else:
                if not has_hands:
                    candidate_label = ""
                stable_count = 0

            if cooldown_count > 0:
                cooldown_count -= 1

            now = time.time()
            if current_tokens and (not has_hands) and (now - last_token_time) >= args.sentence_timeout:
                completed_sentences.append(" ".join(current_tokens))
                current_tokens = []
                candidate_label = ""
                stable_count = 0
                cooldown_count = 0
                recent_labels.clear()
                frame_buffer.clear()

            display = draw_predictions(
                frame=frame,
                predictions=predictions,
                candidate_label=(smoothed_label if top_score >= args.threshold else ""),
                candidate_score=(top_score if top_score >= args.threshold else 0.0),
                has_hands=has_hands,
                current_tokens=current_tokens,
                completed_sentences=completed_sentences,
                show_debug=args.show_debug,
            )
            cv2.imshow("Sign Language Subtitle Inference", display)
            if cv2.getWindowProperty("Sign Language Subtitle Inference", cv2.WND_PROP_VISIBLE) < 1:
                break

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
