from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def filter_topk_labels(
    input_csv: Path,
    output_csv: Path,
    label_column: str = "label",
    top_k: int = 10,
    min_samples: int = 0,
) -> None:
    df = pd.read_csv(input_csv)

    if label_column not in df.columns:
        raise ValueError(f"{label_column} column not found in CSV")

    work_df = df[df[label_column].notna()].copy()
    work_df[label_column] = work_df[label_column].astype(str)

    label_counts = work_df[label_column].value_counts()

    print("=== Label Distribution (Top 20) ===")
    print(label_counts.head(20))

    if min_samples > 0:
        label_counts = label_counts[label_counts >= min_samples]

    top_labels = label_counts.head(top_k).index.tolist()

    print("\n=== Selected Labels ===")
    for i, label in enumerate(top_labels, start=1):
        print(f"{i}. {label} ({label_counts[label]})")

    filtered_df = work_df[work_df[label_column].isin(top_labels)].copy()

    print("\n=== Result ===")
    print(f"original samples: {len(df)}")
    print(f"valid samples: {len(work_df)}")
    print(f"filtered samples: {len(filtered_df)}")
    print(f"num classes: {len(top_labels)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved to: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_topk_labels(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        label_column=args.label_column,
        top_k=args.top_k,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()