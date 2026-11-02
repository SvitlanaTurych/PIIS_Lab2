"""Latency: 10 розігрівних + N вимірюваних запитів до /predict.
python scripts/benchmark.py --url http://localhost:8000 --file samples/yes.wav -n 200 --label docker-multistage
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import requests


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, round(p / 100 * (len(v) - 1)))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--file", default="samples/yes.wav")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--label", default="run")
    a = ap.parse_args()

    data = Path(a.file).read_bytes()
    s = requests.Session()

    def call():
        t0 = time.perf_counter()
        r = s.post(f"{a.url}/predict", files={"file": ("a.wav", data, "audio/wav")})
        r.raise_for_status()
        return (time.perf_counter() - t0) * 1000, r.json()["latency_ms"]

    for _ in range(a.warmup):
        call()
    res = [call() for _ in range(a.n)]
    client = [c for c, _ in res]
    server = [x for _, x in res]
    out = {"label": a.label, "n": a.n}
    for name, v in (("server", server), ("client", client)):
        out[name] = {"mean": round(st.mean(v), 2), "p50": round(pct(v, 50), 2), "p95": round(pct(v, 95), 2)}
    print(json.dumps(out, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{a.label}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
