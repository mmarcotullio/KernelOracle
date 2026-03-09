from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from lstm.models.lstm import LSTMConfig, LSTMNextPid
from lstm.utils.data import TraceWindowDataset, Vocab, batch_to_device
from lstm.utils.metrics import top1_accuracy


def make_loader(npz_path: str, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TraceWindowDataset(npz_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )


def train_one_epoch(model: torch.nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        logits = model(batch["pid"], batch["cont"], state=batch["state"])
        loss = torch.nn.functional.cross_entropy(logits, batch["y"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch["y"].shape[0]
        n += batch["y"].shape[0]

    return total_loss / max(1, n)


@torch.no_grad()
def eval_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    acc_sum = 0.0
    n = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        logits = model(batch["pid"], batch["cont"], state=batch["state"])
        acc = top1_accuracy(logits, batch["y"])
        bsz = batch["y"].shape[0]
        acc_sum += acc * bsz
        n += bsz

    return {"acc": acc_sum / max(1, n)}


@torch.no_grad()
def measure_inference_latency_ms_safe(model: torch.nn.Module,
                                      batch: Dict[str, torch.Tensor],
                                      warmup: int = 20,
                                      iters: int = 100) -> float:
    model.eval()

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(warmup):
        _ = model(batch["pid"], batch["cont"], state=batch["state"])

    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(batch["pid"], batch["cont"], state=batch["state"])
    _sync()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", default="tcn/artifacts/train.npz")
    ap.add_argument("--test_seen_npz", default="tcn/artifacts/test_seen.npz")
    ap.add_argument("--test_unseen_npz", default="tcn/artifacts/test_unseen.npz")
    ap.add_argument("--vocab", default="tcn/artifacts/vocab.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--save_dir", default="tcn/models")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    vocab = Vocab.load(args.vocab)
    cfg = LSTMConfig(
        num_pids=len(vocab.pid_to_idx),
        num_states=len(vocab.state_to_idx),
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model = LSTMNextPid(cfg).to(device)

    train_loader = make_loader(args.train_npz, args.batch_size, shuffle=True)
    seen_loader = make_loader(args.test_seen_npz, args.batch_size, shuffle=False) if os.path.exists(args.test_seen_npz) else None
    unseen_loader = make_loader(args.test_unseen_npz, args.batch_size, shuffle=False) if os.path.exists(args.test_unseen_npz) else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_seen = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device)
        dt = time.time() - t0

        msg = f"epoch={epoch} loss={loss:.4f} time={dt:.1f}s"

        if seen_loader is not None:
            seen = eval_model(model, seen_loader, device)["acc"]
            msg += f" | seen_acc={seen:.4f}"
        else:
            seen = None

        if unseen_loader is not None:
            unseen = eval_model(model, unseen_loader, device)["acc"]
            msg += f" | unseen_acc={unseen:.4f}"

        print(msg)

        if seen is not None and seen > best_seen:
            best_seen = seen
            ckpt_path = os.path.join(args.save_dir, "lstm_nextpid_best.pt")
            torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, ckpt_path)
            print(f"Saved best checkpoint -> {ckpt_path}")

    batch = next(iter(train_loader))
    batch = batch_to_device(batch, device)
    lat_ms = measure_inference_latency_ms_safe(
        model,
        {"pid": batch["pid"], "cont": batch["cont"], "state": batch["state"]},
        warmup=20,
        iters=100
    )
    print(f"Avg forward latency per batch: {lat_ms:.3f} ms (device={device})")


if __name__ == "__main__":
    main()