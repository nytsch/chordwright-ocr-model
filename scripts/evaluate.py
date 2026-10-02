"""
Ein trainiertes Modell prüfen — wie die App es benutzen wird: Strahlsuche mit Grammatik.

    python scripts/evaluate.py runs/base/best.pt --val data/val
    python scripts/evaluate.py chordnet.bin --val data/val     # die exportierte Datei (NumPy-Referenz)

Zählt: richtig gesamt, Akkorde richtig, Nicht-Akkorde erkannt, Vorzeichen
richtig (Akkorde mit ♯/♭), und die häufigsten Verwechslungen.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chordocr.data import load, to_input  # noqa: E402
from chordocr.decode import read_chord  # noqa: E402
from chordocr.export import load_checkpoint  # noqa: E402
from chordocr.infer_numpy import ChordNetFile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--val", type=Path, default=Path("data/val"))
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    images, labels = load(args.val, args.limit)
    if args.model.suffix == ".bin":
        net = ChordNetFile(args.model)
        outputs = [net.log_probs(image) for image in images]
    else:
        model = load_checkpoint(args.model)
        with torch.no_grad():
            outputs = [o for start in range(0, len(images), 512) for o in model(to_input(images[start : start + 512])).numpy()]

    counts: Counter = Counter()
    confusions: Counter = Counter()
    for log_probs, label in zip(outputs, labels):
        read, _ = read_chord(log_probs)
        ok = read == label
        counts["all"] += 1
        counts["right"] += ok
        kind = "chord" if label else "negative"
        counts[kind] += 1
        counts[kind + "_right"] += ok
        if any(a in label for a in "#b") and label[1:2] in "#b":
            counts["accidental"] += 1
            counts["accidental_right"] += ok
        if not ok:
            confusions[(label or "∅", read or "∅")] += 1

    pct = lambda a, b: f"{100 * counts[a] / max(1, counts[b]):.2f} %"  # noqa: E731
    print(f"Gesamt richtig:        {pct('right', 'all')}  ({counts['all']} Bilder)")
    print(f"Akkorde richtig:       {pct('chord_right', 'chord')}")
    print(f"  mit ♯/♭ am Grundton: {pct('accidental_right', 'accidental')}")
    print(f"Kein Akkord erkannt:   {pct('negative_right', 'negative')}")
    print("Häufigste Fehler (soll → gelesen):")
    for (want, got), n in confusions.most_common(15):
        print(f"  {want:>12} → {got:<12} {n}")


if __name__ == "__main__":
    main()
