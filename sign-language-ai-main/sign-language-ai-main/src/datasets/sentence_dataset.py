from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(self, manifest_path: str | Path, landmark_root: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.landmark_root = Path(landmark_root)
        self.records = self._load_records()

        if not self.records:
            raise ValueError("No usable sentence samples found. Check manifest and landmark_root.")

        vocab = sorted({label for record in self.records for label in record["gloss_sequence"]})
        self.blank_idx = 0
        self.label2idx = {label: idx + 1 for idx, label in enumerate(vocab)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.num_classes = len(self.label2idx) + 1

        print(f"usable sentences: {len(self.records)}")
        print(f"num gloss labels: {len(self.label2idx)}")

    def _load_records(self) -> list[dict]:
        records: list[dict] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sentence_id = record.get("sentence_id")
                gloss_sequence = record.get("gloss_sequence") or []
                if not sentence_id or not gloss_sequence:
                    continue

                landmark_path = self.landmark_root / f"{sentence_id}.npy"
                if not landmark_path.exists():
                    continue

                record["landmark_path"] = str(landmark_path)
                records.append(record)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        record = self.records[idx]
        arr = np.load(record["landmark_path"])
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D landmark array, got shape={arr.shape}")

        labels = [self.label2idx[str(label)] for label in record["gloss_sequence"]]
        return {
            "frames": torch.tensor(arr, dtype=torch.float32),
            "frame_length": arr.shape[0],
            "targets": torch.tensor(labels, dtype=torch.long),
            "target_length": len(labels),
            "sentence_id": record["sentence_id"],
            "korean_text": record.get("korean_text") or "",
            "gloss_sequence": record["gloss_sequence"],
        }


def collate_sentence_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    batch = sorted(batch, key=lambda item: int(item["frame_length"]), reverse=True)
    max_frames = max(int(item["frame_length"]) for item in batch)
    input_size = batch[0]["frames"].shape[1]

    padded_frames = []
    input_lengths = []
    flat_targets = []
    target_lengths = []
    sentence_ids = []
    korean_texts = []
    gloss_sequences = []

    for item in batch:
        frames = item["frames"]
        frame_length = int(item["frame_length"])
        pad_len = max_frames - frame_length
        if pad_len > 0:
            pad = torch.zeros((pad_len, input_size), dtype=torch.float32)
            frames = torch.cat([frames, pad], dim=0)

        padded_frames.append(frames)
        input_lengths.append(frame_length)
        flat_targets.append(item["targets"])
        target_lengths.append(int(item["target_length"]))
        sentence_ids.append(str(item["sentence_id"]))
        korean_texts.append(str(item["korean_text"]))
        gloss_sequences.append(list(item["gloss_sequence"]))

    return {
        "frames": torch.stack(padded_frames, dim=0),
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "targets": torch.cat(flat_targets, dim=0),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "sentence_ids": sentence_ids,
        "korean_texts": korean_texts,
        "gloss_sequences": gloss_sequences,
    }
