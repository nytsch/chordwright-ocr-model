"""
Die Referenz für die App: `chordnet.bin` lesen und rechnen, nur mit NumPy.

Jede Schicht hier ist eine Schleife, die in TypeScript genauso aussieht. Liefert
die App für dasselbe Bild dieselben Werte (bis auf Rundung), ist sie richtig.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .export import MAGIC


class ChordNetFile:
    def __init__(self, path: Path):
        data = Path(path).read_bytes()
        assert data[:8] == MAGIC, "keine chordnet.bin"
        (length,) = struct.unpack("<I", data[8:12])
        self.header = json.loads(data[12 : 12 + length])
        self.weights = np.frombuffer(data[12 + length :], dtype="<f2").astype(np.float32)

    def _take(self, offset: int, shape: tuple[int, ...]) -> np.ndarray:
        return self.weights[offset : offset + int(np.prod(shape))].reshape(shape)

    def logits(self, gray: np.ndarray) -> np.ndarray:
        """gray: (32, 192) uint8, 255 = Papier → (steps, classes) Rohwerte."""
        x = (1.0 - gray.astype(np.float32) / 255.0)[None]  # (1, H, W)
        for layer in self.header["layers"]:
            op = layer["op"]
            if op == "conv2d":
                kh, kw = layer["kernel"]
                ph, pw = layer["padding"]
                w = self._take(layer["weight"], (layer["out"], layer["in"], kh, kw))
                b = self._take(layer["bias"], (layer["out"],))
                padded = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
                h, wd = padded.shape[1] - kh + 1, padded.shape[2] - kw + 1
                out = np.zeros((layer["out"], h, wd), dtype=np.float32)
                for dy in range(kh):
                    for dx in range(kw):
                        out += np.einsum("oc,chw->ohw", w[:, :, dy, dx], padded[:, dy : dy + h, dx : dx + wd])
                x = out + b[:, None, None]
                if layer["relu"]:
                    x = np.maximum(x, 0)
            elif op == "maxpool2d":
                kh, kw = layer["kernel"]
                c, h, wd = x.shape
                x = x[:, : h // kh * kh, : wd // kw * kw].reshape(c, h // kh, kh, wd // kw, kw).max(axis=(2, 4))
            elif op == "squeeze":
                x = x[:, 0, :]
            elif op == "conv1d":
                k, d, p = layer["kernel"], layer["dilation"], layer["padding"]
                w = self._take(layer["weight"], (layer["out"], layer["in"], k))
                b = self._take(layer["bias"], (layer["out"],))
                padded = np.pad(x, ((0, 0), (p, p)))
                t = x.shape[1]
                out = np.zeros((layer["out"], t), dtype=np.float32)
                for j in range(k):
                    out += w[:, :, j] @ padded[:, j * d : j * d + t]
                out += b[:, None]
                if layer["relu"]:
                    out = np.maximum(out, 0)
                x = x + out if layer["residual"] else out
        return x.T

    def log_probs(self, gray: np.ndarray) -> np.ndarray:
        z = self.logits(gray)
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))
