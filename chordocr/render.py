"""
Ein Akkordzeichen als Bild — so, wie die App es aus einer Seite schneidet.

Die App (`app/src/ocr/recognize.ts`) findet jedes Zeichen über dem Notensystem
als Kasten um die Tinte, nachdem die Notenlinien weggemalt sind, und reicht den
Ausschnitt in Graustufen weiter. Genau das wird hier nachgestellt:

1. Setzen in großer Auflösung — Grundton, Vorzeichen, Rest, Bass, jedes in
   einer der Arten, wie Blätter es drucken (♯ aus einer Musikschrift, hoch-
   gestellte Zahlen, kleinerer Bass …).
2. Störungen, die an echten Kästen hängen: Notenhals oder Taktstrich daneben,
   Segno/Coda dicht dran, Reste von Linien, Punkte.
3. Knapp um die Tinte schneiden, auf die Größe eines echten Kastens bringen
   (12–48 px hoch) und verschlechtern: Unschärfe, Papier, Rauschen, JPEG.
4. `normalize` — dieselbe Abbildung, die die App vor dem Modell macht.

Ein Teil der Bilder sind Nicht-Akkorde (Zeichen, Wörter, Ziffern) mit leerer
Beschriftung: das Modell soll auch sagen können, dass da kein Akkord steht.
"""
from __future__ import annotations

import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .grammar import sample_chord, split_chord

ROOT = Path(__file__).resolve().parent.parent

HEIGHT = 32
WIDTH = 192

SHARP, FLAT = "♯", "♭"
SMUFL_SHARP, SMUFL_FLAT = "", ""
SEGNO, CODA = "", ""
QUARTER = ""

NEGATIVE_WORDS = [
    "N.C.", "To Coda", "D.S. al Coda", "D.C. al Fine", "Fine", "x2", "2x", "(x3)", "1.", "2.", "3.",
    "rit.", "a tempo", "Intro", "Coda", "mf", "Vamp", "Tag", "Last time", "%", "|", "Fill", "Gtr.",
]


@dataclass
class Font:
    path: Path
    role: str
    split: str
    cmap: set[int]
    axes: list[dict]

    def has(self, text: str) -> bool:
        return all(ord(c) in self.cmap for c in text)


def load_catalogue(split: str, fonts_dir: Path = ROOT / "fonts") -> tuple[list[Font], list[Font]]:
    """Text- und Musikschriften für `split` ("train" oder "val"). Musikschriften gelten für beide."""
    text, music = [], []
    for entry in json.loads((ROOT / "fonts.json").read_text())["fonts"]:
        path = fonts_dir / entry["file"]
        if not path.exists():
            continue
        font_split = entry.get("split", "train")
        if entry["role"] == "text" and font_split != split:
            continue
        cmap = set(TTFont(path, lazy=True).getBestCmap() or {})
        axes = ImageFont.truetype(str(path), 20).get_variation_axes() if _is_variable(path) else []
        (text if entry["role"] == "text" else music).append(Font(path, entry["role"], font_split, cmap, axes))
    if not text or not music:
        raise SystemExit(f"Keine Schriften für '{split}' in {fonts_dir} — zuerst: python scripts/fetch_fonts.py")
    return text, music


def _is_variable(path: Path) -> bool:
    return "fvar" in TTFont(path, lazy=True)


class Renderer:
    def __init__(self, split: str = "train", negative_rate: float = 0.1):
        self.text_fonts, self.music_fonts = load_catalogue(split)
        self.negative_rate = negative_rate
        self._cache: dict[tuple[Path, int], ImageFont.FreeTypeFont] = {}

    # — Schriften —

    def _font(self, font: Font, size: int, weight: float | None = None) -> ImageFont.FreeTypeFont:
        key = (font.path, size)
        if key not in self._cache:
            self._cache[key] = ImageFont.truetype(str(font.path), size)
        face = self._cache[key]
        if font.axes:
            values = []
            for axis in font.axes:
                name = axis.get("name", b"")
                name = name.decode() if isinstance(name, bytes) else str(name)
                if weight is not None and name.lower() in ("weight", "wght"):
                    values.append(axis["minimum"] + weight * (axis["maximum"] - axis["minimum"]))
                else:
                    values.append(axis["default"])
            face.set_variation_by_axes(values)
        return face

    # — eine Probe —

    def sample(self, rng: random.Random) -> tuple[np.ndarray, int, str]:
        """(Bild HEIGHT × WIDTH uint8, belegte Breite, Beschriftung)."""
        size = rng.randint(56, 96)
        canvas = Image.new("L", (size * 17, size * 3), 255)
        draw = ImageDraw.Draw(canvas)
        baseline = int(size * 2.0)
        x0 = size * 2  # Platz links für Hals oder Segno davor

        if rng.random() < self.negative_rate:
            label = ""
            self._negative(rng, draw, x0, baseline, size)
        else:
            label = sample_chord(rng)
            end = self._chord(rng, draw, label, x0, baseline, size)
            self._clutter(rng, draw, x0, end, baseline, size)

        if _ink_box(canvas) is None:  # nichts gesetzt (Zeichen fehlt in der Schrift) — neu würfeln
            return self.sample(rng)
        image = self._degrade(rng, canvas, size)
        array, width = normalize(image, rng)
        return array, width, label

    # — Setzen —

    def _chord(self, rng: random.Random, draw: ImageDraw.ImageDraw, chord: str, x: float, baseline: int, size: int) -> float:
        text_font = rng.choice(self.text_fonts)
        music_font = rng.choice(self.music_fonts)
        weight = rng.random() if text_font.axes else None
        tracking = size * rng.uniform(-0.02, 0.08)

        accidental_style = rng.choices(["ascii", "unicode", "music", "ascii-raised"], weights=[35, 15, 40, 10])[0]
        if accidental_style == "unicode" and not text_font.has(SHARP + FLAT):
            accidental_style = "music"
        superscript = rng.random() < 0.4
        small_bass = rng.random() < 0.25
        accidental_scale = rng.uniform(0.7, 1.0)
        accidental_raise = rng.uniform(0.0, 0.2)
        super_scale = rng.uniform(0.55, 0.8)
        super_raise = rng.uniform(0.25, 0.45)

        def text(s: str, scale: float = 1.0, lift: float = 0.0) -> None:
            nonlocal x
            face = self._font(text_font, max(8, round(size * scale)), weight)
            for ch in s:
                draw.text((x, baseline - lift * size), ch, font=face, fill=0, anchor="ls")
                x += face.getlength(ch) + tracking * scale

        def accidental(a: str, scale: float = 1.0, lift: float = 0.0) -> None:
            nonlocal x
            if accidental_style == "ascii":
                return text(a, scale, lift)
            if accidental_style == "ascii-raised":
                return text(a, scale * 0.75, lift + 0.3)
            if accidental_style == "unicode":
                return text(SHARP if a == "#" else FLAT, scale, lift)
            face = self._font(music_font, max(8, round(size * scale * accidental_scale * 1.6)))
            glyph = SMUFL_SHARP if a == "#" else SMUFL_FLAT
            draw.text((x + size * 0.03, baseline - (lift + accidental_raise + (0.3 if a == '#' else 0.0)) * size), glyph, font=face, fill=0, anchor="ls")
            x += face.getlength(glyph) + size * 0.06

        def run(s: str, scale: float = 1.0, lift: float = 0.0) -> None:
            """Text mit Vorzeichen darin (b9, #5): die Vorzeichen in der gewählten Art."""
            i = 0
            while i < len(s):
                ch = s[i]
                # Ein 'b' vor einer Ziffer ist ein Vorzeichen, sonst ein Buchstabe (sus, add …).
                if ch == "#" or (ch == "b" and i + 1 < len(s) and s[i + 1].isdigit()):
                    accidental(ch, scale, lift)
                else:
                    text(ch, scale, lift)
                i += 1

        root, acc, rest, bass = split_chord(chord)
        text(root)
        if acc:
            accidental(acc)
        if rest:
            if superscript:
                minor = ""
                for prefix in ("min", "m"):
                    if rest.startswith(prefix) and not rest.startswith("maj"):
                        minor = prefix
                        break
                run(minor)
                run(rest[len(minor):], super_scale, super_raise)
            else:
                run(rest)
        if bass:
            scale = 0.75 if small_bass else 1.0
            text("/", scale)
            text(bass[1], scale)
            if len(bass) > 2:
                accidental(bass[2], scale)
        return x

    def _clutter(self, rng: random.Random, draw: ImageDraw.ImageDraw, x0: float, x1: float, baseline: int, size: int) -> None:
        cap = size * 0.72
        if rng.random() < 0.06:  # Notenhals oder Taktstrich, der in den Kasten ragt
            width = max(2, round(size * rng.uniform(0.04, 0.12)))
            gap = size * rng.uniform(0.0, 0.12)
            x = x0 - gap - width if rng.random() < 0.5 else x1 + gap
            top = baseline - cap * rng.uniform(0.2, 1.4)
            draw.rectangle([x, top, x + width, top + cap * rng.uniform(0.6, 1.6)], fill=0)
        if rng.random() < 0.04:  # Segno oder Coda dicht daneben
            face = self._font(rng.choice(self.music_fonts), round(size * rng.uniform(1.0, 1.6)))
            glyph = rng.choice([SEGNO, CODA])
            gap = size * rng.uniform(0.05, 0.3)
            if rng.random() < 0.5:
                draw.text((x1 + gap, baseline), glyph, font=face, fill=0, anchor="ls")
            else:
                draw.text((x0 - gap - face.getlength(glyph), baseline), glyph, font=face, fill=0, anchor="ls")
        if rng.random() < 0.04:  # Rest einer Linie (Klammer, Notenlinie, Hilfslinie)
            y = baseline + size * rng.uniform(-0.05, 0.25)
            a = rng.uniform(x0 - size, x1)
            draw.rectangle([a, y, a + size * rng.uniform(0.3, 3), y + max(1, size * 0.03)], fill=0)
        if rng.random() < 0.03:  # Punkte, Flecken
            for _ in range(rng.randint(1, 3)):
                cx, cy = rng.uniform(x0, x1), baseline + size * rng.uniform(-0.9, 0.2)
                r = size * rng.uniform(0.02, 0.06)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)

    def _negative(self, rng: random.Random, draw: ImageDraw.ImageDraw, x: float, baseline: int, size: int) -> None:
        kind = rng.choices(["sign", "word", "tempo", "key", "stem"], weights=[3, 5, 1, 4, 2])[0]
        if kind == "key":
            # Die Spitze einer Vorzeichnung: ♯/♭ im Zickzack, unten abgeschnitten —
            # so ragt sie aus dem System in den Streifen, in dem die App Akkorde sucht.
            face = self._font(rng.choice(self.music_fonts), round(size * rng.uniform(1.4, 2.2)))
            glyph = rng.choice([SMUFL_SHARP, SMUFL_FLAT])
            step = face.getlength(glyph) * rng.uniform(0.9, 1.2)
            for i in range(rng.randint(1, 4)):
                draw.text((x + i * step, baseline + (size * 0.35 if i % 2 else 0)), glyph, font=face, fill=0, anchor="ls")
            cut = baseline - size * rng.uniform(-0.2, 0.9)
            draw.rectangle([0, cut, x + 6 * step + size, cut + size * 4], fill=255)
        elif kind == "stem":
            width = max(2, round(size * rng.uniform(0.04, 0.1)))
            draw.rectangle([x, baseline - size * rng.uniform(0.4, 1.4), x + width, baseline], fill=0)
            if rng.random() < 0.5:  # mit Fähnchen
                draw.polygon([(x + width, baseline - size), (x + width + size * 0.5, baseline - size * 0.5), (x + width, baseline - size * 0.7)], fill=0)
        elif kind == "sign":
            face = self._font(rng.choice(self.music_fonts), round(size * rng.uniform(1.0, 1.8)))
            draw.text((x, baseline), rng.choice([SEGNO, CODA]), font=face, fill=0, anchor="ls")
        elif kind == "word":
            font = rng.choice(self.text_fonts)
            face = self._font(font, size, rng.random() if font.axes else None)
            draw.text((x, baseline), rng.choice(NEGATIVE_WORDS), font=face, fill=0, anchor="ls")
        else:
            music = [f for f in self.music_fonts if f.has(QUARTER)] or self.music_fonts
            note = self._font(rng.choice(music), round(size * 1.4))
            draw.text((x, baseline), QUARTER, font=note, fill=0, anchor="ls")
            font = rng.choice(self.text_fonts)
            face = self._font(font, size)
            draw.text((x + size * 0.7, baseline), f" = {rng.randint(60, 160)}", font=face, fill=0, anchor="ls")

    # — Verschlechtern —

    def _degrade(self, rng: random.Random, canvas: Image.Image, size: int) -> Image.Image:
        if rng.random() < 0.5:
            canvas = canvas.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, fillcolor=255, expand=False)
        if rng.random() < 0.15:
            shear = rng.uniform(-0.15, 0.15)
            canvas = canvas.transform(canvas.size, Image.AFFINE, (1, shear, -shear * canvas.height / 2, 0, 1, 0), Image.BICUBIC, fillcolor=255)
        if rng.random() < 0.25:  # dünner oder fetter gedruckt
            canvas = canvas.filter(ImageFilter.MinFilter(3) if rng.random() < 0.5 else ImageFilter.MaxFilter(3))

        box = _ink_box(canvas)
        if box is None:
            return Image.new("L", (8, 8), 255)
        margin = [round(size * rng.uniform(0.0, 0.06)) for _ in range(4)]
        x0, y0, x1, y1 = box
        canvas = canvas.crop((max(0, x0 - margin[0]), max(0, y0 - margin[1]), x1 + 1 + margin[2], y1 + 1 + margin[3]))

        # Auf die Größe eines echten Kastens: ein Zeichen ist auf einer Seite
        # 12 bis 48 Pixel hoch, je nach Auflösung und Schriftgröße.
        target = rng.uniform(12, 48)
        scale = target / canvas.height
        canvas = canvas.resize((max(4, round(canvas.width * scale)), max(4, round(target))), _resample(rng))

        if rng.random() < 0.5:
            canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))
        copied = rng.random() < 0.12
        if copied:  # Fax/Kopierer: nur schwarz und weiß — vor Papier und Rauschen, ohne Schatten
            threshold = _otsu(np.asarray(canvas)) + rng.uniform(-20, 20)
            canvas = canvas.point(lambda v: 0 if v < threshold else 255)
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        # Tinte bleibt immer deutlich dunkler als das Papier, auch im Schatten.
        paper = rng.uniform(165, 255)
        shadow = rng.uniform(0, 50) if not copied and rng.random() < 0.3 else 0.0
        ink = rng.uniform(0, min(100, paper - shadow - 70))
        if shadow:  # Schatten: Papier verläuft
            along = 1 if rng.random() < 0.5 else 0
            ramp = np.linspace(0, 1, array.shape[along])
            paper = paper - shadow * (ramp[None, :] if along == 1 else ramp[:, None])
        array = ink + array * (paper - ink)
        if rng.random() < 0.5:
            array = array + np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(2, 14), array.shape)
        canvas = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        if rng.random() < 0.5:
            buffer = io.BytesIO()
            canvas.save(buffer, "JPEG", quality=rng.randint(25, 95))
            canvas = Image.open(buffer).convert("L")
        return canvas


def _otsu(array: np.ndarray) -> float:
    """Die Schwelle, die Tinte und Papier am saubersten trennt."""
    histogram = np.bincount(array.ravel(), minlength=256).astype(np.float64)
    levels = np.arange(256)
    weight = np.cumsum(histogram)
    mean = np.cumsum(histogram * levels)
    total, total_mean = weight[-1], mean[-1]
    between = (total_mean * weight - mean * total) ** 2 / np.maximum(weight * (total - weight), 1e-9)
    return float(np.argmax(between))


def _resample(rng: random.Random) -> int:
    return rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX])


def _ink_box(image: Image.Image, threshold: int = 128) -> tuple[int, int, int, int] | None:
    array = np.asarray(image) < threshold
    ys, xs = np.nonzero(array)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def normalize(image: Image.Image, rng: random.Random | None = None) -> tuple[np.ndarray, int]:
    """
    Der Ausschnitt so, wie das Modell ihn sieht: HEIGHT hoch, Seitenverhältnis
    bleibt, links bündig auf weißem Grund bis WIDTH — breitere werden gestaucht.
    **Die App muss genau das nachbauen** (Graustufen, 255 = Papier).
    """
    width = max(1, min(WIDTH, round(image.width * HEIGHT / max(1, image.height))))
    resample = _resample(rng) if rng else Image.BILINEAR
    scaled = image.convert("L").resize((width, HEIGHT), resample)
    out = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    out[:, :width] = np.asarray(scaled)
    return out, width
