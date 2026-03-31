from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def sanitize_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    return name.strip("_") or "dataset"


def build_mp4_map(video_root: Path) -> dict[str, str]:
    mp4_map: dict[str, str] = {}
    for path in video_root.rglob("*.mp4"):
        mp4_map[path.stem] = str(path)
    return mp4_map


def choose_video_path(base: str, mp4_map: dict[str, str], prefer_view: str = "right") -> str | None:
    if prefer_view == "right":
        return mp4_map.get(f"{base}R") or mp4_map.get(base) or mp4_map.get(f"{base}L")
    if prefer_view == "left":
        return mp4_map.get(f"{base}L") or mp4_map.get(base) or mp4_map.get(f"{base}R")
    return mp4_map.get(base) or mp4_map.get(f"{base}R") or mp4_map.get(f"{base}L")


def build_record_from_json(
    json_path: Path,
    mp4_map: dict[str, str],
    dataset_name: str,
    prefer_view: str = "right",
) -> dict[str, Any] | None:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    sent_info = data.get("krlgg_sntenc") or {}
    sign_info = data.get("sign_script") or {}
    gestures = sign_info.get("sign_gestures_strong") or []

    segments: list[dict[str, Any]] = []
    gloss_sequence: list[str] = []

    for gesture in gestures:
        if not isinstance(gesture, dict):
            continue

        start_sec = gesture.get("start")
        end_sec = gesture.get("end")
        gloss_id = gesture.get("gloss_id")
        if start_sec is None or end_sec is None or gloss_id is None:
            continue

        label = str(gloss_id)
        gloss_sequence.append(label)
        segments.append(
            {
                "label": label,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
            }
        )

    if not segments:
        return None

    base = str(data.get("vido_file_nm") or data.get("id") or json_path.stem).strip()
    video_path = choose_video_path(base, mp4_map, prefer_view=prefer_view)

    sentence_start = min(segment["start_sec"] for segment in segments)
    sentence_end = max(segment["end_sec"] for segment in segments)
    sentence_id = f"{dataset_name}_{base}"

    return {
        "sentence_id": sentence_id,
        "dataset_name": dataset_name,
        "source_id": data.get("id"),
        "json_path": str(json_path),
        "video_base": base,
        "video_path": video_path,
        "view": prefer_view,
        "sentence_start_sec": sentence_start,
        "sentence_end_sec": sentence_end,
        "korean_text": sent_info.get("koreanText"),
        "category": sent_info.get("category"),
        "realm": sent_info.get("realm"),
        "thema": sent_info.get("thema"),
        "detailThema": sent_info.get("detailThema"),
        "gloss_sequence": gloss_sequence,
        "segments": segments,
        "num_segments": len(segments),
    }


def build_sentence_manifest(
    video_root: Path,
    label_root: Path,
    output_jsonl: Path,
    dataset_name: str,
    prefer_view: str = "right",
) -> list[dict[str, Any]]:
    dataset_name = sanitize_name(dataset_name)
    mp4_map = build_mp4_map(video_root)
    json_files = sorted(label_root.rglob("*.json"))

    print(f"found json files: {len(json_files)}")
    print(f"found mp4 files: {len(mp4_map)}")
    print(f"dataset_name: {dataset_name}")

    records: list[dict[str, Any]] = []
    for json_path in json_files:
        try:
            record = build_record_from_json(
                json_path=json_path,
                mp4_map=mp4_map,
                dataset_name=dataset_name,
                prefer_view=prefer_view,
            )
            if record is not None:
                records.append(record)
        except Exception as exc:
            print(f"[WARN] failed to parse {json_path}: {exc}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    matched_video = sum(1 for record in records if record.get("video_path") and Path(record["video_path"]).exists())
    print("\n=== Sentence Manifest Summary ===")
    print(f"total sentences: {len(records)}")
    print(f"sentences with matched video: {matched_video}")
    print(f"sentences without matched video: {len(records) - matched_video}")
    print(f"total gloss tokens: {sum(record['num_segments'] for record in records)}")
    print(f"unique gloss labels: {len({label for record in records for label in record['gloss_sequence']})}")

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--label_root", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, default="default")
    parser.add_argument("--prefer_view", type=str, default="right", choices=["center", "left", "right"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_sentence_manifest(
        video_root=args.video_root,
        label_root=args.label_root,
        output_jsonl=args.output_jsonl,
        dataset_name=args.dataset_name,
        prefer_view=args.prefer_view,
    )
    print(f"\nsentence manifest saved: {args.output_jsonl}")
    print(f"rows: {len(records)}")


if __name__ == "__main__":
    main()
