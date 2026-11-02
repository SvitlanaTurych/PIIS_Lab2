# ================= Stage 1: trainer =================
FROM python:3.10-slim AS trainer
ARG EPOCHS=10
ARG SEED=42
ARG MODEL_VERSION=1.0.0
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src/ ./src/
# Завантаження датасету, навчання, оцінка -> /build/artifacts
RUN python -m src.train --classes yes no up down --epochs ${EPOCHS} --seed ${SEED} \
        --model-version ${MODEL_VERSION} --data-root /build/data --out /build/artifacts

# ================= Stage 2: runtime =================
FROM python:3.10-slim AS runtime
ARG MODEL_VERSION=1.0.0
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="kws-api" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY --from=trainer /build/artifacts ./artifacts

ENV MODEL_PATH=/app/artifacts/model.pt \
    PORT=8000 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
