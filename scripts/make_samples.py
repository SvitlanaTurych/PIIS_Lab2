"""Копіює по одному тестовому .wav на клас у samples/ (запустити один раз і закомітити).
python scripts/make_samples.py --data-root data
"""
import argparse
import shutil
from pathlib import Path

import torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data")
ap.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down"])
a = ap.parse_args()

ds = torchaudio.datasets.SPEECHCOMMANDS(a.data_root, download=True, subset="testing")
done = set()
Path("samples").mkdir(exist_ok=True)
for p in ds._walker:
    c = Path(p).parent.name
    if c in a.classes and c not in done:
        shutil.copy(p, f"samples/{c}.wav")
        done.add(c)
        print("saved", f"samples/{c}.wav")
    if len(done) == len(a.classes):
        break
