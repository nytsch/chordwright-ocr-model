"""
Aus den Wahrscheinlichkeiten je Schritt ein Akkord — oder keiner.

Nicht „beste Zeichenfolge, dann reparieren", sondern: die besten Kandidaten
per CTC-Strahlsuche, und davon der wahrscheinlichste, der ein gültiger Akkord
ist (`grammar.is_chord_symbol`). Gewinnt die leere Lesung, steht da kein Akkord.
Die Sicherheit ist der Anteil des gewählten Akkords an allen Kandidaten — sie
füttert später die Prüfansicht (Phase 1).

Die App baut genau das in TypeScript nach; diese Fassung ist die Referenz.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .grammar import CHARSET, is_chord_symbol


def greedy(log_probs: np.ndarray) -> str:
    best = log_probs.argmax(-1)
    out, previous = [], 0
    for k in best:
        if k != previous and k != 0:
            out.append(CHARSET[k - 1])
        previous = k
    return "".join(out)


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam(log_probs: np.ndarray, width: int = 12, prune: float = -12.0) -> list[tuple[str, float]]:
    """CTC-Präfix-Strahlsuche: (Zeichenfolge, Log-Wahrscheinlichkeit), beste zuerst."""
    # Präfix → (log P endet auf Leer, log P endet auf Zeichen)
    beams: dict[str, tuple[float, float]] = {"": (0.0, -math.inf)}
    for step in log_probs:
        candidates = np.nonzero(step > prune)[0]
        nxt: dict[str, list[float]] = defaultdict(lambda: [-math.inf, -math.inf])
        for prefix, (p_blank, p_char) in beams.items():
            total = _logsumexp(p_blank, p_char)
            for k in candidates:
                p = float(step[k])
                if k == 0:
                    entry = nxt[prefix]
                    entry[0] = _logsumexp(entry[0], total + p)
                    continue
                ch = CHARSET[k - 1]
                extended = prefix + ch
                entry = nxt[extended]
                if prefix and prefix[-1] == ch:
                    # Gleiches Zeichen zweimal nur über ein Leer dazwischen.
                    entry[1] = _logsumexp(entry[1], p_blank + p)
                    same = nxt[prefix]
                    same[1] = _logsumexp(same[1], p_char + p)
                else:
                    entry[1] = _logsumexp(entry[1], total + p)
        ranked = sorted(nxt.items(), key=lambda kv: -_logsumexp(*kv[1]))[:width]
        beams = {k: (v[0], v[1]) for k, v in ranked}
    out = [(k, _logsumexp(*v)) for k, v in beams.items()]
    return sorted(out, key=lambda kv: -kv[1])


def read_chord(log_probs: np.ndarray, width: int = 12) -> tuple[str, float]:
    """(Akkord oder "", Sicherheit 0..1)."""
    candidates = beam(log_probs, width)
    total = -math.inf
    for _, p in candidates:
        total = _logsumexp(total, p)
    for text, p in candidates:
        if text == "" or is_chord_symbol(text):
            return text, math.exp(p - total)
    return "", 0.0
