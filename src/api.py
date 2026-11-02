"""FastAPI-сервіс інференсу. Конфігурація через MODEL_PATH, PORT, LOG_LEVEL, MODEL_VERSION."""
# ruff: noqa: B008
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from .data import to_mono_16k, waveform_to_features
from .model import KWSCNN

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("kws-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.pt"))
MAX_BYTES = 5 * 1024 * 1024
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель завантажується один раз, на старті
    art = MODEL_PATH.parent
    classes = json.loads((art / "labels.json").read_text())
    metrics = json.loads((art / "metrics.json").read_text())
    model = KWSCNN(len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    state.update(
        model=model,
        classes=classes,
        metrics=metrics,
        version=os.getenv("MODEL_VERSION") or metrics.get("model_version", "unknown"),
        size_kb=round(MODEL_PATH.stat().st_size / 1024, 1),
        params=sum(p.numel() for p in model.parameters()),
    )
    log.info("Model loaded: %s (%s KB)", MODEL_PATH, state["size_kb"])
    yield
    state.clear()


app = FastAPI(title="KWS API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    m = state["metrics"]
    return {
        "architecture": m.get("architecture"),
        "classes": state["classes"],
        "parameters": state["params"],
        "model_size_kb": state["size_kb"],
        "accuracy": m.get("accuracy"),  # %
        "model_version": state["version"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Порожній файл")
    if len(raw) > MAX_BYTES:
        raise HTTPException(400, "Файл завеликий (максимум 5 МБ)")
    try:
        wave, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except (sf.LibsndfileError, ValueError) as e:
        raise HTTPException(400, f"Некоректний аудіофайл (очікується .wav): {e}")

    t0 = time.perf_counter()
    try:
        feats = waveform_to_features(to_mono_16k(wave, sr)).unsqueeze(0)
        with torch.inference_mode():
            probs = torch.softmax(state["model"](feats), dim=1)[0]
    except (RuntimeError, ValueError, TypeError) as e:
        raise HTTPException(400, f"Не вдалося обробити аудіо: {e}")
    latency_ms = (time.perf_counter() - t0) * 1000

    classes = state["classes"]
    i = int(probs.argmax())
    return {
        "label": classes[i],
        "confidence": round(float(probs[i]), 3),
        "probabilities": {c: round(float(p), 3) for c, p in zip(classes, probs)},
        "latency_ms": round(latency_ms, 2),
        "model_version": state["version"],
    }
