# chordwright-ocr-model

Akkordzeichen-Erkennung für [Chordwright](https://github.com/nytsch/chordwright):
ein kleines Modell, das ein ausgeschnittenes Akkordzeichen von einem Leadsheet
liest (`F#m7`, `Bb/D`, `A2(no3)` …). Es ersetzt in der App das Raten mit
Tesseract — vor allem beim ♯/♭, das Tesseract gern verschluckt. Hintergrund
und Plan: `docs/NOTENERKENNUNG-PLAN.md` im App-Repo (Phasen 2 und 3).

Die App pinnt dieses Repo per Commit, wie den Datenserver.

## Stand

| Phase | | Stand |
|---|---|---|
| 2 | Synthetische Trainingsdaten | **steht** — `scripts/generate.py` |
| 3 | CRNN + CTC, Training auf der CPU | offen |
| 3 | Export für den Browser (≤ 2–5 MB, iPhone 12) | offen |

## Einrichten

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_fonts.py      # freie Schriften nach fonts/ (nicht im Git)
.venv/bin/python -m pytest -q
```

## Daten erzeugen

```sh
.venv/bin/python scripts/generate.py --preview preview.png               # 120 Proben ansehen
.venv/bin/python scripts/generate.py --count 200000 --out data/train
.venv/bin/python scripts/generate.py --count 10000 --split val --seed 7 --out data/val
```

Etwa 80 Bilder je Sekunde und CPU-Kern, rund 1,5 kB je Bild. Ausgabe:
`part-NNNN.npz` mit `images` (n × 32 × 192, uint8, 255 = Papier), `widths`
und `labels` (leer = kein Akkord). Gleicher Seed, gleiche Daten.

### Was in den Bildern steckt

- **Akkorde** aus der Grammatik der App (`chordocr/grammar.py` = `CHORD_SYMBOL`
  in `app/src/domain/ocrChord.ts`), gewichtet nach Häufigkeit auf Blättern.
- **Satz:** 33 Textschriften, ♯/♭ als `#`/`b`, als Unicode-Zeichen oder aus
  einer Musikschrift (Bravura, Petaluma, Leland, MuseJazz); Erweiterungen
  hochgestellt oder auf der Linie; kleinerer Bass.
- **Störungen:** Notenhals/Taktstrich am Rand, Segno/Coda dicht daneben, Linienreste, Flecken.
- **Verschlechterung:** Schräge, Scherung, dünner/fetter Druck, 12–48 px Höhe,
  Unschärfe, graues Papier, Schatten, Rauschen, JPEG, Schwarz-Weiß-Kopie.
- **Nicht-Akkorde** (6 %): Segno, Coda, `N.C.`, `To Coda`, `x2`, `♩ = 75` … mit
  leerer Beschriftung, damit das Modell „kein Akkord" sagen kann.
- **Prüfschriften:** Die Schriften mit `"split": "val"` in `fonts.json` kommen
  nur in die Prüfmenge — so zeigt sich, ob das Modell auf fremde Schriften
  verallgemeinert. Echte Blätter aus dem Korpus der App prüfen zusätzlich, sie
  werden nicht zum Training benutzt.

### Vertrag mit der App

`chordocr/render.py` → `normalize`: Graustufen-Ausschnitt um die Tinte, auf
32 px Höhe skaliert (Seitenverhältnis bleibt), links bündig auf Weiß bis
192 px, breitere gestaucht. Die App muss genau das vor dem Modell nachbauen.
Handschrift ist bewusst nicht dabei.

## Lizenzen

Code: MIT. Schriften: SIL Open Font License, sie liegen nicht im Repo, sondern
werden von `fetch_fonts.py` aus den Quellen in `fonts.json` geholt.
