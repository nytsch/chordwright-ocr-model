"""
Prüfdateien für den TypeScript-Nachbau in der App: ein winziges Modell
(gleiche Bauart, ein Viertel der Kanäle), ein Eingabebild und was die
NumPy-Referenz daraus rechnet — alles in einer JSON-Datei.

    python scripts/make_fixture.py ../chordwright/app/src/domain/__fixtures__
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chordocr.decode import beam  # noqa: E402
from chordocr.export import export  # noqa: E402
from chordocr.infer_numpy import ChordNetFile  # noqa: E402
from chordocr.model import ChordNet  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
model = ChordNet(width=0.25)
model.train()
with torch.no_grad():
    for _ in range(3):
        model(torch.rand(16, 1, 32, 192))
export(model.eval(), out / "chordnet-tiny.bin")
net = ChordNetFile(out / "chordnet-tiny.bin")
gray = (np.random.default_rng(1).random((32, 192)) * 255).astype(np.uint8)
log_probs = net.log_probs(gray)
(out / "chordnet-tiny.json").write_text(json.dumps({
    "model": base64.b64encode((out / "chordnet-tiny.bin").read_bytes()).decode(),
    "gray": base64.b64encode(gray.tobytes()).decode(),
    "logProbs": np.round(log_probs, 5).tolist(),
    "beam": [[t, round(p, 4)] for t, p in beam(log_probs)[:5]],
}))
(out / "chordnet-tiny.bin").unlink()  # steckt base64 im JSON — die App-Tests laden nur Text
print((out / "chordnet-tiny.json").stat().st_size, "Byte")
