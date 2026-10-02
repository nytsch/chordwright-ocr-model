"""
Synthetische Akkordbilder erzeugen.

    python scripts/generate.py --count 200000 --out data/train
    python scripts/generate.py --count 10000 --split val --seed 7 --out data/val
    python scripts/generate.py --preview preview.png      # 120 Proben als Bogen ansehen

Ausgabe: Scheiben `part-0000.npz` mit `images` (n × 32 × 192, uint8, 255 =
Papier), `widths` (belegte Breite) und `labels` (Zeichenketten, leer = kein Akkord).
Gleicher Seed, gleiche Daten.
"""
from __future__ import annotations

import argparse
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chordocr.render import HEIGHT, WIDTH, Renderer  # noqa: E402

SHARD = 20000


def _shard(job: tuple[int, int, str, int, str]) -> str:
    index, count, split, seed, out = job
    renderer = Renderer(split)
    rng = random.Random(seed * 1_000_003 + index)
    images = np.empty((count, HEIGHT, WIDTH), dtype=np.uint8)
    widths = np.empty(count, dtype=np.int16)
    labels = []
    for i in range(count):
        images[i], widths[i], label = renderer.sample(rng)
        labels.append(label)
    path = Path(out) / f"part-{index:04d}.npz"
    np.savez_compressed(path, images=images, widths=widths, labels=np.array(labels))
    return str(path)


def preview(path: Path, split: str, seed: int, count: int = 120) -> None:
    renderer = Renderer(split)
    rng = random.Random(seed)
    columns, cell_h = 4, HEIGHT + 16
    sheet = Image.new("L", (columns * (WIDTH + 8), (count // columns) * cell_h), 200)
    draw = ImageDraw.Draw(sheet)
    for i in range(count):
        image, _, label = renderer.sample(rng)
        x, y = (i % columns) * (WIDTH + 8), (i // columns) * cell_h
        sheet.paste(Image.fromarray(image), (x, y))
        draw.text((x + 2, y + HEIGHT + 1), label or "(kein Akkord)", fill=0)
    sheet.save(path)
    print(f"Bogen: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    if args.preview:
        return preview(args.preview, args.split, args.seed)
    out = args.out or Path("data") / args.split
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, start in enumerate(range(0, args.count, SHARD)):
        jobs.append((index, min(SHARD, args.count - start), args.split, args.seed, str(out)))
    with Pool(args.workers) as pool:
        for path in pool.imap_unordered(_shard, jobs):
            print(path, flush=True)


if __name__ == "__main__":
    main()
