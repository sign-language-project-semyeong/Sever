from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.models.ctc_model import CTCSignEncoder
from src.preprocess.extract_landmarks import HandLandmarkExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints_sequence/best_ctc_model.pt"))
    parser.add_argument("--model_asset_path", type=Path, default=Path("models/mediapipe/hand_landmarker.task"))
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=180)
    parser.add_argument("--decode_interval", type=int, default=5)
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()


def load_font(size: int):
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CTCSignEncoder(
        input_size=int(checkpoint.get("input_size", 126)),
        hidden_size=int(checkpoint.get("hidden_size", 192)),
        num_layers=int(checkpoint.get("num_layers", 2)),
        num_classes=int(checkpoint["num_classes"]),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx2label = {int(k): str(v) for k, v in checkpoint["idx2label"].items()}
    blank_idx = int(checkpoint.get("blank_idx", 0))
    input_size = int(checkpoint.get("input_size", 126))
    return model, idx2label, blank_idx, input_size


def greedy_decode(logits: torch.Tensor, blank_idx: int, idx2label: dict[int, str]) -> list[str]:
    token_ids = logits.argmax(dim=-1)[0].cpu().tolist()
    collapsed: list[int] = []
    prev_id = None
    for token_id in token_ids:
        if token_id == blank_idx:
            prev_id = token_id
            continue
        if token_id != prev_id:
            collapsed.append(token_id)
        prev_id = token_id
    return [idx2label[token_id] for token_id in collapsed if token_id in idx2label]


def draw_overlay(frame: np.ndarray, decoded_glosses: list[str], frame_count: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    title_font = load_font(30)
    body_font = load_font(24)

    gloss_text = " ".join(decoded_glosses) if decoded_glosses else "???"
    draw.text((20, 20), f"Gloss: {gloss_text}", font=title_font, fill=(0, 255, 0))
    draw.text((20, 60), f"Frames buffered: {frame_count}", font=body_font, fill=(255, 255, 0))
    draw.text((20, frame.shape[0] - 40), "q: quit | c: clear buffer", font=body_font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx2label, blank_idx, input_size = load_model(args.checkpoint, device)
    extractor = HandLandmarkExtractor(max_num_hands=2, model_asset_path=str(args.model_asset_path))
    frame_buffer: deque[np.ndarray] = deque(maxlen=args.buffer_size)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        extractor.close()
        raise ValueError(f"Failed to open camera index: {args.camera_index}")

    decoded_glosses: list[str] = []
    frame_index = 0

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
            frame_buffer.append(frame_feature)
            frame_index += 1

            if frame_buffer and frame_index % args.decode_interval == 0:
                sequence = np.stack(list(frame_buffer), axis=0).astype(np.float32)
                sequence_tensor = torch.tensor(sequence[None, :, :], dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits = model(sequence_tensor)
                decoded_glosses = greedy_decode(logits, blank_idx, idx2label)

            display = draw_overlay(frame, decoded_glosses, len(frame_buffer))
            cv2.imshow("Sign Language Realtime Sequence Inference", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                frame_buffer.clear()
                decoded_glosses = []
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    main()
