from __future__ import annotations
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.preprocess.extract_landmarks import HandLandmarkExtractor
_EXTRACTOR: HandLandmarkExtractor | None = None
_MODEL_ASSET_PATH: str | None = None
def load_records(manifest_path: Path) -> list[dict]:
    records: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
def _init_worker(model_asset_path: str) -> None:
    global _EXTRACTOR, _MODEL_ASSET_PATH
    _MODEL_ASSET_PATH = model_asset_path
    _EXTRACTOR = HandLandmarkExtractor(max_num_hands=2, model_asset_path=model_asset_path)
def _ensure_extractor() -> HandLandmarkExtractor:
    global _EXTRACTOR, _MODEL_ASSET_PATH
    if _EXTRACTOR is None:
        if not _MODEL_ASSET_PATH:
            raise ValueError("model asset path is not initialized")
        _EXTRACTOR = HandLandmarkExtractor(max_num_hands=2, model_asset_path=_MODEL_ASSET_PATH)
    return _EXTRACTOR
def _process_record(task: tuple[dict, str, bool]) -> tuple[str, str, str]:
    record, save_root_str, overwrite = task
    sentence_id = record.get("sentence_id")
    video_path = record.get("video_path")
    start_sec = record.get("sentence_start_sec")
    end_sec = record.get("sentence_end_sec")
    if not sentence_id or not video_path or start_sec is None or end_sec is None:
        return ("skipped_missing", str(sentence_id or "unknown"), "missing fields")
    video_path = str(video_path)
    if not Path(video_path).exists():
        return ("skipped_missing", str(sentence_id), "video missing")
    save_root = Path(save_root_str)
    save_path = save_root / f"{sentence_id}.npy"
    if save_path.exists() and not overwrite:
        return ("skipped_exists", str(sentence_id), "already exists")
    try:
        extractor = _ensure_extractor()
        arr = extractor.extract_from_video_segment(
            video_path=video_path,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            max_frames=None,
        )
        np.save(save_path, arr)
        return ("saved", str(sentence_id), "")
    except Exception as exc:
        return ("failed", str(sentence_id), str(exc))
def process_sentence_manifest(
    manifest_path: Path,
    save_root: Path,
    model_asset_path: Path,
    overwrite: bool = False,
    num_workers: int = 1,
) -> None:
    records = load_records(manifest_path)
    save_root.mkdir(parents=True, exist_ok=True)
    tasks = [(record, str(save_root), overwrite) for record in records]
    status_counts = {
        "saved": 0,
        "skipped_exists": 0,
        "skipped_missing": 0,
        "failed": 0,
    }
    if num_workers <= 1:
        _init_worker(str(model_asset_path))
        result_iter = map(_process_record, tasks)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(str(model_asset_path),))
        result_iter = pool.imap_unordered(_process_record, tasks)
    try:
        for status, sentence_id, message in tqdm(result_iter, total=len(tasks), desc="extract_sentence_landmarks"):
            status_counts[status] += 1
            if status == "failed":
                print(f"[WARN] failed on {sentence_id}: {message}")
    finally:
        if num_workers > 1:
            pool.close()
            pool.join()
        elif _EXTRACTOR is not None:
            _EXTRACTOR.close()
    print("\n=== Sentence Extraction Summary ===")
    print(f"saved: {status_counts['saved']}")
    print(f"skipped(existing): {status_counts['skipped_exists']}")
    print(f"skipped(missing): {status_counts['skipped_missing']}")
    print(f"failed: {status_counts['failed']}")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--save_root", type=Path, required=True)
    parser.add_argument("--model_asset_path", type=Path, default=Path("models/mediapipe/hand_landmarker.task"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=max(1, (mp.cpu_count() or 1) // 2))
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    process_sentence_manifest(
        manifest_path=args.manifest_path,
        save_root=args.save_root,
        model_asset_path=args.model_asset_path,
        overwrite=args.overwrite,
        num_workers=max(1, args.num_workers),
    )
if __name__ == "__main__":
    main()
