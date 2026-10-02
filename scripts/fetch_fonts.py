"""Lädt die Schriften aus fonts.json nach fonts/ — einmal vor dem Erzeugen der Daten."""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "fonts"


def main() -> int:
    FONTS.mkdir(exist_ok=True)
    catalogue = json.loads((ROOT / "fonts.json").read_text())["fonts"]
    failed = 0
    for font in catalogue:
        target = FONTS / font["file"]
        if target.exists():
            continue
        try:
            with urllib.request.urlopen(font["url"], timeout=60) as response:
                target.write_bytes(response.read())
            print(f"  {font['file']}")
        except Exception as error:  # noqa: BLE001 — eine fehlende Schrift soll die anderen nicht aufhalten
            failed += 1
            print(f"  {font['file']}: {error}", file=sys.stderr)
    print(f"{len(catalogue) - failed}/{len(catalogue)} Schriften in {FONTS}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
