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
| 3 | Faltungsnetz + CTC, Training auf der CPU | **steht** — `scripts/train.py` |
| 3 | Export für den Browser (≤ 2–5 MB, iPhone 12) | **steht** — `scripts/export.py`, ~0,6 MB |
| 3 | Nachbau in der App (TypeScript) | **steht** — `app/src/domain/chordNet.ts`, gleiche Werte wie die NumPy-Referenz |

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

## Trainieren, prüfen, exportieren

```sh
.venv/bin/python scripts/train.py --epochs 12 --out runs/base          # CPU, Abbrechen + --resume geht
.venv/bin/python scripts/evaluate.py runs/base/best.pt                   # wie die App: Strahlsuche + Grammatik
.venv/bin/python scripts/export.py runs/base/best.pt --out chordnet.bin  # Datei für die App, gegen PyTorch geprüft
.venv/bin/python scripts/evaluate.py chordnet.bin                        # dieselbe Prüfung mit der NumPy-Referenz
```

**Modell** (`chordocr/model.py`): fünf 3×3-Faltungen mit Max-Pooling, eine
Faltung, die die Höhe auflöst, drei gedehnte 1D-Faltungen mit Restverbindung,
eine 1×1-Faltung auf 36 Klassen (35 Zeichen + Leer), 48 Zeitschritte. Rund
290.000 Gewichte, als float16 knapp 0,6 MB. Bewusst ohne LSTM: nur Faltung,
ReLU, Pooling, Addition — das lässt sich in der App in wenigen Zeilen
TypeScript nachbauen, ohne Laufzeit-Paket (onnxruntime-web allein wäre größer
als das Budget).

**Lesen** (`chordocr/decode.py`): CTC-Strahlsuche, dann der wahrscheinlichste
Kandidat, der ein gültiger Akkord ist — oder „kein Akkord". Die Sicherheit
geht später an die Prüfansicht.

**Datei** (`chordocr/export.py`): `chordnet.bin` = Kennung, JSON-Kopf mit den
Schichten (BatchNorm eingerechnet), float16-Gewichte. `chordocr/infer_numpy.py`
rechnet sie nur mit NumPy — die Referenz, an der der TypeScript-Nachbau
gemessen wird.

### Vertrag mit der App

`chordocr/render.py` → `normalize`: Graustufen-Ausschnitt um die Tinte, auf
32 px Höhe skaliert (Seitenverhältnis bleibt), links bündig auf Weiß bis
192 px, breitere gestaucht. Die App muss genau das vor dem Modell nachbauen.
Handschrift ist bewusst nicht dabei.

## Stand des Modells (`release/chordnet.bin`)

Die App pinnt diese Datei per Commit und Prüfsumme (`chordwrightOcrModel` in
`app/package.json`).

**Echte Blätter** (40 Leadsheets aus `nytsch/chordwright-corpus`, rund 2.180
Akkorde, je Symbol an seiner Stelle geprüft, `npm run ocr:bench` in der App, an
demselben Stand der App gemessen):

| | Modell (v6) | Modell (v5) |
|---|---|---|
| richtig | **94,6 %** | 94,4 % |
| falsch gelesen | 10 | 20 |
| nicht gefunden | 109 | 100 |
| zu viel (kein Akkord) | 34 | 32 |

v5 las deutsch gesetzte Akkorde (`Hm`, `D/Fis`) gar nicht; v6 liest sie (Vater
unser von DMMK: 25 von 25). Die App nimmt ein alleinstehendes `H` nur auf einem
Blatt, das auch sonst deutsch setzt — sonst ist es ein Strich, den das Modell
für den Buchstaben hielt.

**Synthetisch**, Schriften, die das Training nie sah (`scripts/evaluate.py`):
99,2 % Akkorde, 98,9 % mit ♯/♭, 99,4 % Nicht-Akkorde erkannt.

Trainiert: 8 Epochen auf 200.000 Bildern, dann viermal je 3 Epochen
nachtrainiert (`--init`), jedes Mal mit Nicht-Akkorden, die das Messen an echten
Blättern gezeigt hat: Liedtext, Vorzeichnungen, Schlüsselenden, Taktzahlen
(auch über der Schlüsselspitze), gerahmte Abschnitts-Kästchen. v6: aus der
v5-Datei (`--init release/chordnet.bin`, BatchNorm eingefroren) 4 Epochen auf
150.000 Bildern, davon 12 % deutsch gesetzt (`H`, `Fis`, `Es` …), Zeichensatz
um `H` und `e` erweitert.

## Lizenzen

- **Code und Modell** (`release/chordnet.bin`): MIT, © 2026 Niko Krämer — siehe
  `LICENSE`. Die App führt das Modell in ihrer `NOTICE`.
- **Trainingsdaten:** ausschließlich künstlich erzeugt (`scripts/generate.py`).
  Keine echten Notenblätter, keine Liedtexte aus Liederbüchern — der Korpus
  echter Blätter (privat) dient nur zum Prüfen und ist nie ins Training gegangen.
- **Schriften:** alle unter der SIL Open Font License 1.1 (Bravura, Petaluma,
  Leland, Edwin, MuseJazz, Google Fonts — Quellen in `fonts.json`). Sie liegen
  nicht im Repo, `fetch_fonts.py` holt sie. Die OFL erlaubt, mit den Schriften
  Bilder zu setzen; die Bilder und das daraus gelernte Modell sind keine
  Schriftsoftware und fallen nicht unter die OFL.
- **Werkzeuge zum Trainieren** (nicht Teil des Modells): NumPy, PyTorch (BSD),
  Pillow (HPND), fontTools (MIT).
