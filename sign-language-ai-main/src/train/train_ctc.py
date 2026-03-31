from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.datasets.sentence_dataset import SentenceDataset, collate_sentence_batch
from src.models.ctc_model import CTCSignEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def greedy_decode(logits: torch.Tensor, input_lengths: torch.Tensor, blank_idx: int, idx2label: dict[int, str]) -> list[list[str]]:
    pred_ids = logits.argmax(dim=-1).cpu()
    sequences: list[list[str]] = []

    for batch_idx in range(pred_ids.size(0)):
        length = int(input_lengths[batch_idx].item())
        raw_ids = pred_ids[batch_idx, :length].tolist()
        collapsed: list[int] = []
        prev_id = None
        for token_id in raw_ids:
            if token_id == blank_idx:
                prev_id = token_id
                continue
            if token_id != prev_id:
                collapsed.append(token_id)
            prev_id = token_id
        sequences.append([idx2label[token_id] for token_id in collapsed if token_id in idx2label])

    return sequences


def compute_sequence_accuracy(predictions: list[list[str]], targets: list[list[str]]) -> float:
    if not predictions:
        return 0.0
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    return correct / len(predictions)


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_samples = 0
    all_predictions: list[list[str]] = []
    all_targets: list[list[str]] = []
    idx2label = loader.dataset.dataset.idx2label if isinstance(loader.dataset, Subset) else loader.dataset.idx2label
    blank_idx = loader.dataset.dataset.blank_idx if isinstance(loader.dataset, Subset) else loader.dataset.blank_idx

    for batch in loader:
        frames = batch["frames"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(frames)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = frames.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = greedy_decode(logits.detach(), batch["input_lengths"], blank_idx, idx2label)
        all_predictions.extend(predictions)
        all_targets.extend(batch["gloss_sequences"])

    avg_loss = total_loss / max(total_samples, 1)
    seq_acc = compute_sequence_accuracy(all_predictions, all_targets)
    return avg_loss, seq_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--landmark_root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/checkpoints_sequence"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"seed: {args.seed}")

    dataset = SentenceDataset(
        manifest_path=args.manifest_path,
        landmark_root=args.landmark_root,
    )

    if len(dataset) < 2:
        raise ValueError("Dataset is too small to split into train/val.")

    lengths = [len(record["gloss_sequence"]) for record in dataset.records]
    indices = np.arange(len(dataset))

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=lengths,
        )
    except ValueError as exc:
        print("[WARN] stratified split failed. Falling back to random split.")
        print(f"[WARN] reason: {exc}")
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_ratio,
            random_state=args.seed,
            shuffle=True,
        )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sentence_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sentence_batch)

    model = CTCSignEncoder(
        input_size=126,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=dataset.num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CTCLoss(blank=dataset.blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    print(f"train sentences: {len(train_ds)}")
    print(f"val sentences: {len(val_ds)}")
    print(f"num gloss labels: {len(dataset.label2idx)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_seq_acc = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_seq_acc = run_one_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_seq_acc={train_seq_acc:.4f} "
            f"val_loss={val_loss:.4f} val_seq_acc={val_seq_acc:.4f}"
        )

        if val_seq_acc >= best_val_acc:
            best_val_acc = val_seq_acc
            ckpt_path = args.checkpoint_dir / "best_ctc_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2idx": dataset.label2idx,
                    "idx2label": dataset.idx2label,
                    "blank_idx": dataset.blank_idx,
                    "input_size": 126,
                    "num_classes": dataset.num_classes,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "seed": args.seed,
                    "best_val_seq_acc": best_val_acc,
                },
                ckpt_path,
            )
            print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
