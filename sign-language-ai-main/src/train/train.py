from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.datasets.sign_dataset import SignDataset
from src.models.gru_model import GRUSignClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_non_blocking: bool = False,
    grad_clip_norm: float | None = None,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=use_non_blocking)
        y = y.to(device, non_blocking=use_non_blocking)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def eval_one_epoch(model, loader, criterion, device, use_non_blocking: bool = False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=use_non_blocking)
            y = y.to(device, non_blocking=use_non_blocking)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--landmark_root", type=Path, required=True)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"seed: {args.seed}")

    dataset = SignDataset(
        manifest_csv=args.manifest_csv,
        landmark_root=args.landmark_root,
        max_len=args.max_len,
        text_label_column=args.label_column,
    )

    if len(dataset) < 2:
        raise ValueError("Dataset is too small to split into train/val.")

    input_size = 126
    num_classes = len(dataset.label2idx)

    labels = [
        dataset.label2idx[str(dataset.df.iloc[i][args.label_column])]
        for i in range(len(dataset))
    ]
    indices = np.arange(len(dataset))

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=labels,
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

    print(f"train samples: {len(train_ds)}")
    print(f"val samples: {len(val_ds)}")
    print(f"num classes: {num_classes}")

    use_pin_memory = device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": max(0, args.num_workers),
        "pin_memory": use_pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = GRUSignClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_non_blocking=use_pin_memory,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device, use_non_blocking=use_pin_memory
        )
        scheduler.step(val_acc)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = args.checkpoint_dir / "best_gru_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2idx": dataset.label2idx,
                    "idx2label": dataset.idx2label,
                    "input_size": input_size,
                    "num_classes": num_classes,
                    "max_len": args.max_len,
                    "label_column": args.label_column,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "seed": args.seed,
                    "best_val_acc": best_val_acc,
                },
                ckpt_path,
            )
            print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
