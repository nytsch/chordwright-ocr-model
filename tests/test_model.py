import numpy as np
import torch

from chordocr.data import to_input
from chordocr.decode import beam, greedy, read_chord
from chordocr.export import export
from chordocr.grammar import CHARSET
from chordocr.infer_numpy import ChordNetFile
from chordocr.model import CLASSES, STEPS, ChordNet


def _trained_looking_model() -> ChordNet:
    torch.manual_seed(0)
    model = ChordNet()
    # BatchNorm mit echten Statistiken, damit das Einrechnen etwas zu tun hat.
    model.train()
    with torch.no_grad():
        for _ in range(3):
            model(torch.rand(16, 1, 32, 192))
    return model.eval()


def test_export_matches_torch(tmp_path):
    model = _trained_looking_model()
    export(model, tmp_path / "chordnet.bin")
    net = ChordNetFile(tmp_path / "chordnet.bin")
    gray = (np.random.default_rng(1).random((32, 192)) * 255).astype(np.uint8)
    with torch.no_grad():
        expected = model(to_input(gray[None]))[0].numpy()
    got = net.log_probs(gray)
    assert got.shape == (STEPS, CLASSES)
    assert np.abs(got - expected).max() < 0.05  # float16-Gewichte


def _path(text: str) -> np.ndarray:
    """Log-Wahrscheinlichkeiten, die `text` mit Leer dazwischen sehr sicher sagen."""
    lp = np.full((STEPS, CLASSES), -20.0)
    lp[:, 0] = 0.0
    for i, ch in enumerate(text):
        lp[2 * i + 1, :] = -20.0
        lp[2 * i + 1, CHARSET.index(ch) + 1] = 0.0
    return lp


def test_decoders_read_a_clear_path():
    for text in ["F#m7", "Add9", "Bb/D", ""]:
        assert greedy(_path(text)) == text
        assert beam(_path(text))[0][0] == text


def test_read_chord_skips_what_is_not_a_chord():
    lp = _path("F#m")
    # Zweite Wahl an Schritt 3: '#' fast so wahrscheinlich wie 'i' — 'Fim' ist kein Akkord.
    lp[3, CHARSET.index("#") + 1] = -0.9
    lp[3, CHARSET.index("i") + 1] = -0.5
    assert read_chord(lp)[0] == "F#m"
