"""Навчання + оцінка -> artifacts/{model.pt, labels.json, metrics.json}"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import KWSDataset
from .evaluate import accuracy
from .model import KWSCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--model-version", default="1.0.0")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    t0 = time.perf_counter()
    train = KWSDataset(a.data_root, "train", a.classes)
    val = KWSDataset(a.data_root, "val", a.classes)
    test = KWSDataset(a.data_root, "test", a.classes)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    g = torch.Generator().manual_seed(a.seed)
    tl = torch.utils.data.DataLoader(train, batch_size=a.batch_size, shuffle=True, generator=g)
    vl = torch.utils.data.DataLoader(val, batch_size=128)
    el = torch.utils.data.DataLoader(test, batch_size=128)

    model = KWSCNN(len(a.classes))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in tqdm(tl, desc=f"epoch {ep}/{a.epochs}", leave=False):
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        print(f"epoch {ep}: val_acc={accuracy(model, vl):.2f}%", flush=True)

    test_acc = accuracy(model, el)
    print(f"TEST accuracy = {test_acc:.2f}%")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "labels.json").write_text(json.dumps(a.classes))
    (out / "metrics.json").write_text(json.dumps({
        "accuracy": round(test_acc, 2),  # у відсотках
        "test_samples": len(test),
        "epochs": a.epochs,
        "seed": a.seed,
        "model_version": a.model_version,
        "architecture": "KWSCNN: 3x[Conv3x3-BN-ReLU-MaxPool] + GAP + Linear",
        "train_time_s": round(time.perf_counter() - t0, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
