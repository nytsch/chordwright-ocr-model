import json
import random
from pathlib import Path

import pytest

from chordocr.grammar import is_chord_symbol
from chordocr.render import HEIGHT, WIDTH, Renderer, load_catalogue

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "fonts").exists(), reason="erst scripts/fetch_fonts.py")


def test_samples_have_the_shape_the_model_takes():
    renderer = Renderer("train")
    rng = random.Random(3)
    for _ in range(200):
        image, width, label = renderer.sample(rng)
        assert image.shape == (HEIGHT, WIDTH) and image.dtype.name == "uint8"
        assert 1 <= width <= WIDTH
        assert label == "" or is_chord_symbol(label)
        assert image[:, :width].min() < image[:, :width].max()  # es steht etwas da


def test_same_seed_same_data():
    a = Renderer("train").sample(random.Random(11))
    b = Renderer("train").sample(random.Random(11))
    assert (a[0] == b[0]).all() and a[2] == b[2]


def test_validation_fonts_never_reach_training():
    held_out = {f["file"] for f in json.loads((ROOT / "fonts.json").read_text())["fonts"] if f.get("split") == "val"}
    train, _ = load_catalogue("train")
    val, _ = load_catalogue("val")
    assert held_out and not held_out & {f.path.name for f in train}
    assert {f.path.name for f in val} == held_out
