"""
Das Modell: ein kleines, rein faltendes Netz mit CTC.

Bewusst ohne LSTM und ohne Aufmerksamkeit — nur Faltung, ReLU, Max-Pooling und
Addition. Das sind die Bausteine, die sich in der App ohne Laufzeit-Paket in
ein paar Dutzend Zeilen TypeScript nachbauen lassen (onnxruntime-web allein
wäre größer als das ganze Budget). Die Breite des Bildes wird zur Zeitachse:
48 Schritte für 192 Pixel, je Schritt eine Verteilung über `CHARSET` + Leer.

Eingabe: (n, 1, 32, 192), Werte 0..1 mit 1 = Tinte (also `1 - grau/255`).
Ausgabe: (n, 48, len(CHARSET) + 1) Log-Wahrscheinlichkeiten, Index 0 = Leer.
"""
from __future__ import annotations

import torch
from torch import nn

from .grammar import CHARSET

CLASSES = len(CHARSET) + 1
STEPS = 48


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class Residual1d(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.relu(self.norm(self.conv(x)))


class ChordNet(nn.Module):
    def __init__(self, width: float = 1.0):
        super().__init__()
        # `width` < 1 nur für Testdateien: dieselbe Bauart, ein Bruchteil der Gewichte.
        c = [max(4, round(n * width)) for n in (16, 32, 64, 64, 96, 128)]
        self.features = nn.Sequential(
            _block(1, c[0]), nn.MaxPool2d(2),              # 16 × 96
            _block(c[0], c[1]), nn.MaxPool2d(2),           # 8 × 48
            _block(c[1], c[2]), _block(c[2], c[3]),
            nn.MaxPool2d((2, 1)),                          # 4 × 48
            _block(c[3], c[4]), nn.MaxPool2d((2, 1)),      # 2 × 48
            nn.Conv2d(c[4], c[5], (2, 1), bias=False),     # 1 × 48: Höhe aufgelöst
            nn.BatchNorm2d(c[5]), nn.ReLU(inplace=True),
        )
        self.sequence = nn.Sequential(Residual1d(c[5], 1), Residual1d(c[5], 2), Residual1d(c[5], 4))
        self.classify = nn.Conv1d(c[5], CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(2)        # (n, C, 48)
        x = self.classify(self.sequence(x))    # (n, CLASSES, 48)
        return x.permute(0, 2, 1).log_softmax(-1)


def encode(label: str) -> list[int]:
    return [CHARSET.index(ch) + 1 for ch in label]
