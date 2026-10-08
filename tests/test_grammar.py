import random

from chordocr.grammar import CHARSET, is_chord_symbol, sample_chord, split_chord


def test_every_sampled_chord_is_one_the_app_accepts():
    rng = random.Random(1)
    for _ in range(20000):
        chord = sample_chord(rng)
        assert is_chord_symbol(chord), chord
        assert set(chord) <= set(CHARSET), chord


def test_split_keeps_every_character():
    for chord in ["F#m7", "Bb/D", "A2(no3)", "C", "Ebmaj7/G", "Hm", "D/Fis", "Esmaj7/B", "Asus4"]:
        assert "".join(split_chord(chord)) == chord


def test_app_grammar():
    assert is_chord_symbol("F#m7b5/E")
    assert not is_chord_symbol("AmiG")
    assert is_chord_symbol("H7")
    assert is_chord_symbol("Hm")
    assert is_chord_symbol("D/Fis")
    assert is_chord_symbol("Esmaj7/G")
    assert not is_chord_symbol("Bis")
