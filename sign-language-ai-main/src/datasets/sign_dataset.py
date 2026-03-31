from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SignDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        landmark_root: str | Path,
        max_len: int = 30,
        text_label_column: str = "label",
    ) -> None:
        self.df = pd.read_csv(manifest_csv)
        self.landmark_root = Path(landmark_root)
        self.max_len = max_len
        self.text_label_column = text_label_column

        if "sample_id" not in self.df.columns:
            raise ValueError("manifest_csv must contain 'sample_id' column")
        if self.text_label_column not in self.df.columns:
            raise ValueError(f"manifest_csv must contain '{self.text_label_column}' column")

        valid_df = self.df[
            self.df["sample_id"].notna() & self.df[self.text_label_column].notna()
        ].copy()

        valid_df["sample_id"] = valid_df["sample_id"].astype(str)
        valid_df[self.text_label_column] = valid_df[self.text_label_column].astype(str)

        valid_df["landmark_path"] = valid_df["sample_id"].apply(
            lambda sid: str(self.landmark_root / f"{sid}.npy")
        )

        valid_df = valid_df[
            valid_df["landmark_path"].apply(lambda p: Path(p).exists())
        ].copy()

        self.df = valid_df.reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError("No usable samples found. Check manifest and landmark_root.")

        labels = sorted(self.df[self.text_label_column].unique().tolist())
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        print(f"usable samples: {len(self.df)}")
        print(f"num classes: {len(self.label2idx)}")

    def __len__(self) -> int:
        return len(self.df)

    def _pad_or_trim(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array (T, D), got shape={arr.shape}")

        if len(arr) > self.max_len:
            arr = arr[: self.max_len]
        elif len(arr) < self.max_len:
            pad = np.zeros((self.max_len - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])

        return arr.astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        landmark_path = Path(row["landmark_path"])
        label_text = str(row[self.text_label_column])
        label_idx = self.label2idx[label_text]

        arr = np.load(landmark_path)

        if arr.size == 0:
            raise ValueError(f"Empty landmark array: {landmark_path}")

        arr = self._pad_or_trim(arr)

        x = torch.tensor(arr, dtype=torch.float32)
        y = torch.tensor(label_idx, dtype=torch.long)
        return x, y