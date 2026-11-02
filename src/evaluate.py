"""Accuracy (%) на тестовому наборі: python -m src.evaluate --artifacts artifacts"""
import argparse
import json
from pathlib import Path

import torch

from .data import KWSDataset
from .model import KWSCNN


@torch.no_grad()
def accuracy(model, loader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data-root", default="/tmp/data")
    a = ap.parse_args()
    art = Path(a.artifacts)
    classes = json.loads((art / "labels.json").read_text())
    model = KWSCNN(len(classes))
    model.load_state_dict(torch.load(art / "model.pt", weights_only=True))
    ds = KWSDataset(a.data_root, "test", classes)
    loader = torch.utils.data.DataLoader(ds, batch_size=128)
    print(f"accuracy = {accuracy(model, loader):.2f}% on {len(ds)} test samples")


if __name__ == "__main__":
    main()
