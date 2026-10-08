"""
Welche Akkordzeichen es gibt — und wie oft sie auf echten Blättern stehen.

Die Grammatik ist die der App (`app/src/domain/ocrChord.ts`, `CHORD_SYMBOL`).
Sie muss mit ihr übereinstimmen: Was das Modell hier lernt, prüft die App
dort. `tests/test_grammar.py` hält beide Fassungen nebeneinander fest; ändert
sich die Regel in der App, zuerst hier nachziehen.
"""
from __future__ import annotations

import random
import re

# Wörtlich aus app/src/domain/ocrChord.ts (Stand Oktober 2026).
# Ein Ton, englisch (F#, Bb) oder deutsch gesetzt (Fis, Es, H) — DMMK und andere
# deutsche Verlage schreiben „Hm" und „D/Fis".
NOTE = r"(?:[A-G][#b]?|H|[CDFGA]is|Des|Es|Ges|As)"
CHORD_SYMBOL = re.compile(
    rf"^{NOTE}(?:(?:maj|min|dim|aug|sus|add|no|m|M)[0-9]*|[0-9]+|[#b][0-9]+)*(?:\([a-z0-9#b]+\))?(?:\/{NOTE})?$"
)

# Alles, was in einer Beschriftung vorkommen kann. Index 0 ist im Modell das
# CTC-Leerzeichen; die Zeichen beginnen bei 1.
CHARSET = "ABCDEFGHMabdegijmnosu#/()0123456789"


def is_chord_symbol(text: str) -> bool:
    return CHORD_SYMBOL.match(text) is not None


ROOTS = "CDEFGAB"
# Vorzeichen, die auf Blättern praktisch nicht vorkommen.
RARE = {"E#", "B#", "Fb", "Cb"}

# Endungen mit grobem Gewicht — Lobpreis- und Pop-Leadsheets zuerst, Jazz dahinter.
SUFFIXES: list[tuple[str, float]] = [
    ("", 34), ("m", 18), ("7", 7), ("m7", 6), ("2", 5), ("sus", 3), ("sus4", 4), ("sus2", 2),
    ("maj7", 3), ("add9", 2), ("add2", 1), ("6", 1), ("m6", 0.5), ("9", 0.8), ("m9", 0.5),
    ("maj9", 0.4), ("11", 0.3), ("m11", 0.3), ("13", 0.3), ("7sus4", 1), ("7sus", 0.4),
    ("dim", 0.6), ("dim7", 0.4), ("aug", 0.4), ("5", 0.6), ("M7", 0.3), ("m7b5", 0.5),
    ("7b9", 0.3), ("7#9", 0.3), ("7#5", 0.2), ("7b5", 0.2), ("9sus4", 0.2), ("madd9", 0.4),
    ("min", 0.2), ("min7", 0.2), ("4", 0.6), ("2(no3)", 0.3), ("(add9)", 0.2), ("m(add9)", 0.2),
    ("sus2(no3)", 0.05), ("6(add9)", 0.05), ("13(b9)", 0.05), ("7(b9)", 0.1), ("7(#9)", 0.1),
]
_SUFFIX_TEXT = [s for s, _ in SUFFIXES]
_SUFFIX_WEIGHT = [w for _, w in SUFFIXES]


def _note(rng: random.Random) -> str:
    while True:
        root = rng.choice(ROOTS)
        accidental = rng.choices(["", "#", "b"], weights=[70, 15, 15])[0]
        if root + accidental not in RARE:
            return root + accidental


# Deutsch gesetzt: H statt B, B statt Bb, -is/-es statt #/b.
GERMAN = {"B": "H", "Bb": "B", "C#": "Cis", "D#": "Dis", "F#": "Fis", "G#": "Gis", "A#": "Ais",
          "Db": "Des", "Eb": "Es", "Gb": "Ges", "Ab": "As"}
GERMAN_RATE = 0.12


def sample_chord(rng: random.Random) -> str:
    """Ein Akkord, so wie er gedruckt steht."""
    root, bass = _note(rng), _note(rng) if rng.random() < 0.15 else None
    if rng.random() < GERMAN_RATE:
        root, bass = GERMAN.get(root, root), bass and GERMAN.get(bass, bass)
    chord = root + rng.choices(_SUFFIX_TEXT, weights=_SUFFIX_WEIGHT)[0]
    return chord + "/" + bass if bass else chord


def split_chord(chord: str) -> tuple[str, str, str, str]:
    """Grundton, Vorzeichen, Rest, Bass (mit Schrägstrich) — der Renderer setzt sie verschieden."""
    m = re.match(r"^(H|[CDFGA]is|Des|Ges|(?:Es|As)(?!us)|[A-G])([#b]?)(.*?)((?:/(?:H|[CDFGA]is|Des|Es|Ges|As|[A-G][#b]?))?)$", chord)
    assert m, chord
    return m.group(1), m.group(2), m.group(3), m.group(4)
