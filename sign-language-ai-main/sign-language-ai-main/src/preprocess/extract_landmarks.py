from __future__ import annotations

import argparse
import multiprocessing as mp_process
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


class HandLandmarkExtractor:
    def __init__(self, max_num_hands: int = 2, model_asset_path: str | None = None) -> None:
        self.max_num_hands = max_num_hands
        self.model_asset_path = model_asset_path
        self.expected_len = self.max_num_hands * 21 * 3

        self.use_tasks_api = hasattr(mp, "tasks")
        self.mp_hands = None
        self.landmarker = None
        self._timestamp_ms = 0

        if self.use_tasks_api:
            self._init_tasks_landmarker()
        else:
            self.mp_hands = mp.solutions.hands
            # Solutions API: Hands 객체를 매 프레임 생성하면 매우 느림 → 미리 생성해둠
            self._hands_instance = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def _init_tasks_landmarker(self) -> None:
        if not self.model_asset_path:
            raise ValueError("model_asset_path is required when using MediaPipe tasks API")

        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_asset_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _empty_frame(self) -> list[float]:
        return [0.0] * self.expected_len

    def _extract_result_vector_tasks(self, result) -> list[float]:
        if not getattr(result, "hand_landmarks", None):
            return self._empty_frame()

        hands_with_order: list[tuple[int, list[float]]] = []

        for idx, landmarks in enumerate(result.hand_landmarks[: self.max_num_hands]):
            handedness_list = []
            if getattr(result, "handedness", None) and idx < len(result.handedness):
                handedness_list = result.handedness[idx]

            order_key = idx
            if handedness_list:
                label = (handedness_list[0].category_name or "").lower()
                if label == "left":
                    order_key = 0
                elif label == "right":
                    order_key = 1

            frame_vec: list[float] = []
            for lm in landmarks:
                frame_vec.extend([float(lm.x), float(lm.y), float(lm.z)])
            hands_with_order.append((order_key, frame_vec))

        hands_with_order.sort(key=lambda item: item[0])

        merged: list[float] = []
        for _, hand_vec in hands_with_order[: self.max_num_hands]:
            merged.extend(hand_vec)

        if len(merged) < self.expected_len:
            merged.extend([0.0] * (self.expected_len - len(merged)))

        return merged

    def _extract_result_vector_solutions(self, result) -> list[float]:
        frame_vec: list[float] = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks[: self.max_num_hands]:
                for lm in hand_landmarks.landmark:
                    frame_vec.extend([lm.x, lm.y, lm.z])

        if len(frame_vec) < self.expected_len:
            frame_vec.extend([0.0] * (self.expected_len - len(frame_vec)))

        return frame_vec

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        if hasattr(self, "_hands_instance") and self._hands_instance is not None:
            self._hands_instance.close()
            self._hands_instance = None

    def extract_from_frame(self, frame: np.ndarray, fps: float = 30.0) -> np.ndarray:
        if frame is None:
            return np.zeros((1, self.expected_len), dtype=np.float32)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._timestamp_ms += max(1, int(1000 / max(fps, 1.0)))
            result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
            frame_vec = self._extract_result_vector_tasks(result)
        else:
            # 미리 생성된 Hands 인스턴스 재사용 (매 프레임 생성 방지)
            result = self._hands_instance.process(rgb)
            frame_vec = self._extract_result_vector_solutions(result)

        return np.array([frame_vec], dtype=np.float32)

    def extract_from_video_segment(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        max_frames: int | None = None,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        start_frame = max(0, int(start_sec * fps))
        end_frame = max(start_frame + 1, int(end_sec * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        all_frames: list[list[float]] = []

        if self.use_tasks_api:
            frame_count = 0
            current_frame = start_frame

            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame += 1
                frame_count += 1

                if max_frames is not None and frame_count > max_frames:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self._timestamp_ms += max(1, int(1000 / fps))
                timestamp_ms = self._timestamp_ms
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                all_frames.append(self._extract_result_vector_tasks(result))
        else:
            with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                frame_count = 0
                current_frame = start_frame

                while current_frame < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_frame += 1
                    frame_count += 1

                    if max_frames is not None and frame_count > max_frames:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)
                    all_frames.append(self._extract_result_vector_solutions(result))

        cap.release()

        if not all_frames:
            return np.zeros((1, self.expected_len), dtype=np.float32)

        return np.array(all_frames, dtype=np.float32)


_EXTRACTOR: HandLandmarkExtractor | None = None
_MODEL_ASSET_PATH: str | None = None
_MAX_FRAMES: int | None = None


def _init_worker(model_asset_path: str | None, max_frames: int | None) -> None:
    global _EXTRACTOR, _MODEL_ASSET_PATH, _MAX_FRAMES
    _MODEL_ASSET_PATH = model_asset_path
    _MAX_FRAMES = max_frames
    _EXTRACTOR = HandLandmarkExtractor(max_num_hands=2, model_asset_path=model_asset_path)


def _ensure_extractor() -> HandLandmarkExtractor:
    global _EXTRACTOR, _MODEL_ASSET_PATH
    if _EXTRACTOR is None:
        _EXTRACTOR = HandLandmarkExtractor(max_num_hands=2, model_asset_path=_MODEL_ASSET_PATH)
    return _EXTRACTOR


def _process_row(task: tuple[dict, str, bool]) -> tuple[str, str, str]:
    row, save_root_str, overwrite = task
    sample_id = row.get("sample_id")
    video_path = row.get("video_path")
    start_sec = row.get("start_sec")
    end_sec = row.get("end_sec")

    if pd.isna(sample_id) or pd.isna(video_path) or pd.isna(start_sec) or pd.isna(end_sec):
        return ("skipped_missing", str(sample_id or "unknown"), "missing fields")

    video_path = str(video_path)
    if not Path(video_path).exists():
        return ("skipped_missing", str(sample_id), "video missing")

    save_path = Path(save_root_str) / f"{sample_id}.npy"
    if save_path.exists() and not overwrite:
        return ("skipped_exists", str(sample_id), "already exists")

    try:
        extractor = _ensure_extractor()
        arr = extractor.extract_from_video_segment(
            video_path=video_path,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            max_frames=_MAX_FRAMES,
        )
        np.save(save_path, arr)
        return ("saved", str(sample_id), "")
    except Exception as exc:
        return ("failed", str(sample_id), str(exc))


def process_manifest(
    manifest_csv: Path,
    save_root: Path,
    max_frames: int | None = None,
    overwrite: bool = False,
    model_asset_path: Path | None = None,
    num_workers: int = 1,
) -> None:
    df = pd.read_csv(manifest_csv)
    save_root.mkdir(parents=True, exist_ok=True)

    tasks = [(row._asdict() if hasattr(row, '_asdict') else row.to_dict(), str(save_root), overwrite) for _, row in df.iterrows()]
    status_counts = {
        "saved": 0,
        "skipped_exists": 0,
        "skipped_missing": 0,
        "failed": 0,
    }

    if num_workers <= 1:
        _init_worker(str(model_asset_path) if model_asset_path else None, max_frames)
        result_iter = map(_process_row, tasks)
    else:
        ctx = mp_process.get_context("spawn")
        pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(str(model_asset_path) if model_asset_path else None, max_frames),
        )
        result_iter = pool.imap_unordered(_process_row, tasks)

    try:
        for status, sample_id, message in tqdm(result_iter, total=len(tasks), desc="extract_landmarks"):
            status_counts[status] += 1
            if status == "failed":
                print(f"[WARN] failed on {sample_id}: {message}")
    finally:
        if num_workers > 1:
            pool.close()
            pool.join()
        elif _EXTRACTOR is not None:
            _EXTRACTOR.close()

    print("\n=== Extraction Summary ===")
    print(f"saved: {status_counts['saved']}")
    print(f"skipped(existing): {status_counts['skipped_exists']}")
    print(f"skipped(missing): {status_counts['skipped_missing']}")
    print(f"failed: {status_counts['failed']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--save_root", type=Path, required=True)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--model_asset_path",
        type=Path,
        default=Path("models/mediapipe/hand_landmarker.task"),
    )
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_manifest(
        manifest_csv=args.manifest_csv,
        save_root=args.save_root,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        model_asset_path=args.model_asset_path,
        num_workers=max(1, args.num_workers),
    )
    print("landmark extraction complete")


if __name__ == "__main__":
    main()
