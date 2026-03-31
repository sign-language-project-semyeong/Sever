from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def sanitize_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    return name.strip("_") or "dataset"


def build_mp4_map(video_root: Path) -> dict[str, str]:
    mp4_map: dict[str, str] = {}
    for path in video_root.rglob("*.mp4"):
        mp4_map[path.stem] = str(path)
    return mp4_map


def choose_video_path(
    base: str,
    mp4_map: dict[str, str],
    prefer_view: str = "right",
) -> str | None:
    if prefer_view == "right":
        return mp4_map.get(f"{base}R") or mp4_map.get(base) or mp4_map.get(f"{base}L")
    if prefer_view == "left":
        return mp4_map.get(f"{base}L") or mp4_map.get(base) or mp4_map.get(f"{base}R")
    return mp4_map.get(base) or mp4_map.get(f"{base}R") or mp4_map.get(f"{base}L")


def build_rows_from_json(
    json_path: Path,
    mp4_map: dict[str, str],
    dataset_name: str,
    prefer_view: str = "right",
) -> list[dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sent_info = data.get("krlgg_sntenc") or {}
    sign_info = data.get("sign_script") or {}

    base = data.get("vido_file_nm") or data.get("id") or json_path.stem
    base = str(base).strip()
    video_path = choose_video_path(base, mp4_map, prefer_view=prefer_view)

    gestures = sign_info.get("sign_gestures_strong") or []
    rows: list[dict[str, Any]] = []

    for idx, g in enumerate(gestures):
        if not isinstance(g, dict):
            continue

        start_sec = g.get("start")
        end_sec = g.get("end")
        gloss_id = g.get("gloss_id")

        if start_sec is None or end_sec is None or gloss_id is None:
            continue

        sample_id = f"{dataset_name}_{base}_{idx:03d}"

        rows.append(
            {
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "source_id": data.get("id"),
                "json_path": str(json_path),
                "video_base": base,
                "video_path": video_path,
                "view": prefer_view,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "label": str(gloss_id),
                "korean_text": sent_info.get("koreanText"),
                "category": sent_info.get("category"),
                "realm": sent_info.get("realm"),
                "thema": sent_info.get("thema"),
                "detailThema": sent_info.get("detailThema"),
            }
        )

    return rows


def build_manifest(
    video_root: Path,
    label_root: Path,
    output_csv: Path,
    dataset_name: str,
    prefer_view: str = "right",
) -> pd.DataFrame:
    dataset_name = sanitize_name(dataset_name)
    mp4_map = build_mp4_map(video_root)
    rows: list[dict[str, Any]] = []

    json_files = sorted(label_root.rglob("*.json"))
    print(f"found json files: {len(json_files)}")
    print(f"found mp4 files: {len(mp4_map)}")
    print(f"dataset_name: {dataset_name}")

    for json_path in json_files:
        try:
            rows.extend(
                build_rows_from_json(
                    json_path=json_path,
                    mp4_map=mp4_map,
                    dataset_name=dataset_name,
                    prefer_view=prefer_view,
                )
            )
        except Exception as exc:
            print(f"[WARN] failed to parse {json_path}: {exc}")

    df = pd.DataFrame(rows)

    if not df.empty:
        df["video_exists"] = df["video_path"].apply(lambda x: Path(x).exists() if pd.notna(x) else False)

        print("\n=== Manifest Summary ===")
        print(f"total rows: {len(df)}")
        print(f"rows with matched video: {int(df['video_exists'].sum())}")
        print(f"rows without matched video: {int((~df['video_exists']).sum())}")
        print(f"unique labels: {df['label'].nunique() if 'label' in df.columns else 0}")

        duplicated_ids = df["sample_id"].duplicated().sum()
        print(f"duplicated sample_id count: {duplicated_ids}")

        if duplicated_ids > 0:
            dup_df = df[df["sample_id"].duplicated(keep=False)].sort_values("sample_id")
            print("\n[WARN] duplicated sample_id detected:")
            print(dup_df[["sample_id", "json_path", "video_base"]].head(20))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--label_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, default="default")
    parser.add_argument("--prefer_view", type=str, default="right", choices=["center", "left", "right"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_manifest(
        video_root=args.video_root,
        label_root=args.label_root,
        output_csv=args.output_csv,
        dataset_name=args.dataset_name,
        prefer_view=args.prefer_view,
    )
    print(f"\nmanifest saved: {args.output_csv}")
    print(df.head())
    print(f"rows: {len(df)}")


if __name__ == "__main__":
    main()