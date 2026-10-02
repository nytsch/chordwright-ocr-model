# Arbeitsanweisungen für Claude

Gleiche Regeln wie im App-Repo `nytsch/chordwright` (CLAUDE.md dort): erst
lokalisieren, kurz antworten, Scope eng halten.

- Python 3.11, `.venv/`, `pip install -r requirements.txt`, Tests: `.venv/bin/python -m pytest -q`.
- Schriften: `scripts/fetch_fonts.py` → `fonts/` (gitignored). Nur freie Schriften (OFL/Apache), in `fonts.json` mit Quelle.
- `chordocr/grammar.py` spiegelt `CHORD_SYMBOL` aus `app/src/domain/ocrChord.ts` — Änderungen dort hier nachziehen.
- `chordocr/render.py` `normalize` ist der Vertrag mit der App (32 × 192, Graustufen, 255 = Papier). Nie ändern, ohne die App mitzuziehen.
- Echte Blätter nie ins Repo und nie ins Training — nur zum Prüfen.
- Keine Handschrift.
