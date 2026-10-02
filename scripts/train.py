"""
Das Modell trainieren — auf der CPU, aus den Scheiben von generate.py.

    python scripts/train.py --train data/train --val data/val --epochs 12 --out runs/base

Legt in --out ab: `best.pt` (bestes Modell auf der Prüfmenge), `last.pt`,
`metrics.jsonl` (eine Zeile je Epoche). Abbrechen und mit --resume weitermachen geht.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chordocr.data import batch_targets, load, to_input  # noqa: E402
from chordocr.decode import greedy  # noqa: E402
from chordocr.model import STEPS, ChordNet  # noqa: E402


def evaluate(model: ChordNet, images: np.ndarray, labels: list[str], batch: int = 512) -> dict:
    model.eval()
    right = chords = chords_right = negatives = negatives_right = 0
    with torch.no_grad():
        for start in range(0, len(labels), batch):
            out = model(to_input(images[start : start + batch])).numpy()
            for log_probs, label in zip(out, labels[start : start + batch]):
                read = greedy(log_probs)
                right += read == label
                if label:
                    chords += 1
                    chords_right += read == label
                else:
                    negatives += 1
                    negatives_right += read == ""
    return {
        "accuracy": right / len(labels),
        "chords": chords_right / max(1, chords),
        "negatives": negatives_right / max(1, negatives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("data/train"))
    parser.add_argument("--val", type=Path, default=Path("data/val"))
    parser.add_argument("--out", type=Path, default=Path("runs/base"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--limit", type=int, default=None, help="nur so viele Trainingsbilder (zum Ausprobieren)")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init", type=Path, default=None, help="Gewichte eines trainierten Modells als Start (Nachtraining)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    images, labels = load(args.train, args.limit)
    val_images, val_labels = load(args.val)
    print(f"{len(labels)} Trainings-, {len(val_labels)} Prüfbilder, {torch.get_num_threads()} Threads", flush=True)

    model = ChordNet()
    if args.init:
        state = torch.load(args.init, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
    print(f"{sum(p.numel() for p in model.parameters()):,} Parameter", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = (len(labels) + args.batch - 1) // args.batch
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    args.out.mkdir(parents=True, exist_ok=True)
    start_epoch, best = 0, -1.0
    if args.resume and (args.out / "last.pt").exists():
        state = torch.load(args.out / "last.pt", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        start_epoch, best = state["epoch"] + 1, state["best"]

    rng = np.random.default_rng(0)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = rng.permutation(len(labels))
        started, total = time.time(), 0.0
        for step, begin in enumerate(range(0, len(order), args.batch)):
            index = order[begin : begin + args.batch]
            x = to_input(images[index])
            targets, lengths = batch_targets([labels[i] for i in index])
            log_probs = model(x).permute(1, 0, 2)  # (T, n, C) für CTCLoss
            loss = ctc(log_probs, targets, torch.full((len(index),), STEPS, dtype=torch.long), lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            schedule.step()
            total += loss.item()
            if step % 200 == 0:
                print(f"  Epoche {epoch + 1} Schritt {step}/{steps_per_epoch} Verlust {loss.item():.3f}", flush=True)
        metrics = {"epoch": epoch + 1, "loss": total / steps_per_epoch, "seconds": round(time.time() - started), **evaluate(model, val_images, val_labels)}
        print(json.dumps(metrics), flush=True)
        with open(args.out / "metrics.jsonl", "a") as log:
            log.write(json.dumps(metrics) + "\n")
        if metrics["accuracy"] > best:
            best = metrics["accuracy"]
            torch.save(model.state_dict(), args.out / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "epoch": epoch, "best": best}, args.out / "last.pt")


if __name__ == "__main__":
    main()
