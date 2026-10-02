"""
Das trainierte Modell in ein Format, das die App ohne Laufzeit-Paket liest.

Datei `chordnet.bin`:

    8 Byte  "CWOCR\\x00\\x01\\x00"     Kennung, Fassung 1
    4 Byte  Länge des Kopfes (uint32, little endian)
    Kopf    JSON (UTF-8), mit Leerzeichen auf ein Vielfaches von 4 aufgefüllt
    Rest    alle Gewichte als float16, little endian, in der Reihenfolge des Kopfes

Der Kopf beschreibt die Schichten der Reihe nach; BatchNorm ist schon in die
Faltungen eingerechnet. `infer_numpy.py` liest dieselbe Datei — die App muss
dasselbe Ergebnis liefern wie diese Referenz.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .grammar import CHARSET
from .model import STEPS, ChordNet, Residual1d
from .render import HEIGHT, WIDTH

MAGIC = b"CWOCR\x00\x01\x00"


def _fold(conv: nn.Conv1d | nn.Conv2d, norm: nn.BatchNorm1d | nn.BatchNorm2d | None) -> tuple[np.ndarray, np.ndarray]:
    weight = conv.weight.detach().double()
    bias = conv.bias.detach().double() if conv.bias is not None else torch.zeros(weight.shape[0], dtype=torch.float64)
    if norm is not None:
        scale = norm.weight.detach().double() / torch.sqrt(norm.running_var.double() + norm.eps)
        weight = weight * scale.view(-1, *([1] * (weight.dim() - 1)))
        bias = (bias - norm.running_mean.double()) * scale + norm.bias.detach().double()
    return weight.float().numpy(), bias.float().numpy()


def export(model: ChordNet, path: Path) -> dict:
    model.eval()
    layers: list[dict] = []
    blobs: list[np.ndarray] = []
    offset = 0

    def add(weight: np.ndarray, bias: np.ndarray) -> dict:
        nonlocal offset
        entry = {"weight": offset, "bias": offset + weight.size}
        offset += weight.size + bias.size
        blobs.extend([weight.ravel(), bias.ravel()])
        return entry

    modules = list(model.features)
    i = 0
    while i < len(modules):
        m = modules[i]
        if isinstance(m, nn.Sequential):  # Faltung + BatchNorm + ReLU
            conv, norm = m[0], m[1]
            w, b = _fold(conv, norm)
            layers.append({"op": "conv2d", "in": conv.in_channels, "out": conv.out_channels, "kernel": list(conv.kernel_size), "padding": list(conv.padding), "relu": True, **add(w, b)})
        elif isinstance(m, nn.MaxPool2d):
            k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
            layers.append({"op": "maxpool2d", "kernel": list(k)})
        elif isinstance(m, nn.Conv2d):  # die Faltung, die die Höhe auflöst, mit ihrer BatchNorm und ReLU
            w, b = _fold(m, modules[i + 1])
            layers.append({"op": "conv2d", "in": m.in_channels, "out": m.out_channels, "kernel": list(m.kernel_size), "padding": list(m.padding), "relu": True, **add(w, b)})
            i += 2
        i += 1
    layers.append({"op": "squeeze"})  # (C, 1, T) → (C, T)
    for block in model.sequence:
        assert isinstance(block, Residual1d)
        w, b = _fold(block.conv, block.norm)
        layers.append({"op": "conv1d", "in": block.conv.in_channels, "out": block.conv.out_channels, "kernel": block.conv.kernel_size[0], "dilation": block.conv.dilation[0], "padding": block.conv.padding[0], "relu": True, "residual": True, **add(w, b)})
    w, b = _fold(model.classify, None)
    layers.append({"op": "conv1d", "in": model.classify.in_channels, "out": model.classify.out_channels, "kernel": 1, "dilation": 1, "padding": 0, "relu": False, "residual": False, **add(w, b)})

    header = {
        "format": 1,
        "input": {"height": HEIGHT, "width": WIDTH, "ink": "1 - gray / 255"},
        "steps": STEPS,
        "charset": CHARSET,
        "blank": 0,
        "layers": layers,
        "count": offset,
    }
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    weights = np.concatenate(blobs).astype("<f2")
    path.write_bytes(MAGIC + struct.pack("<I", len(text)) + text + weights.tobytes())
    return header


def load_checkpoint(path: Path) -> ChordNet:
    model = ChordNet()
    state = torch.load(path, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.eval()
