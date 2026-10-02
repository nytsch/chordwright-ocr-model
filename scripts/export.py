"""
Das Modell für die App schreiben und gegen PyTorch prüfen.

    python scripts/export.py runs/base/best.pt --out chordnet.bin
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chordocr.data import load, to_input  # noqa: E402
from chordocr.export import export, load_checkpoint  # noqa: E402
from chordocr.infer_numpy import ChordNetFile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("chordnet.bin"))
    parser.add_argument("--val", type=Path, default=Path("data/val"))
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    header = export(model, args.out)
    net = ChordNetFile(args.out)
    images, _ = load(args.val, 50)
    with torch.no_grad():
        expected = model(to_input(images)).numpy()
    worst = max(float(np.abs(net.log_probs(image) - e).max()) for image, e in zip(images, expected))
    same = sum(int((net.log_probs(image).argmax(-1) == e.argmax(-1)).all()) for image, e in zip(images, expected))
    size = args.out.stat().st_size
    print(f"{args.out}: {size / 1024:.0f} KB, {header['count']:,} Gewichte, sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]}")
    print(f"Gegen PyTorch: größte Abweichung {worst:.4f}, gleiche Lesung bei {same}/{len(images)}")


if __name__ == "__main__":
    main()
