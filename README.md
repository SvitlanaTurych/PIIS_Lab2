# Лабораторна робота №2 — Docker для ML-сервісу (KWS на Speech Commands)

Класи: `yes`, `no`, `up`, `down`. CNN над лог-Mel спектрограмою, FastAPI-сервіс, multi-stage Docker, CI в GitHub Actions.

## Структура
```
src/          data.py, model.py, train.py, evaluate.py, api.py
scripts/      benchmark.py, cold_start.sh, local_cold_start.py, make_samples.py
Dockerfile        multi-stage (trainer -> runtime)
Dockerfile.single single-stage (для порівняння розмірів)
.github/workflows/ci.yml
```

## Локальний запуск (venv, без Docker)
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train --epochs 10 --seed 42 --data-root data --out artifacts
python -m src.evaluate --artifacts artifacts --data-root data
python scripts/make_samples.py --data-root data     # один раз, закомітити samples/*.wav
MODEL_PATH=artifacts/model.pt uvicorn src.api:app --port 8000
```

## Docker
```bash
docker build -t kws-api:1.0.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg BUILD_DATE=$(date -u +%FT%TZ) .
docker build -f Dockerfile.single -t kws-api:single .
docker run -d --name kws -p 8000:8000 kws-api:1.0.0
docker ps            # healthy
curl localhost:8000/health
curl localhost:8000/model-info
curl -X POST -F "file=@samples/yes.wav" localhost:8000/predict
```

## Вимірювання
```bash
python scripts/benchmark.py --label docker-multistage -n 200     # latency (сервер/клієнт)
scripts/cold_start.sh kws-api:1.0.0                              # холодний старт + RAM
python scripts/local_cold_start.py                               # те саме без Docker
docker images kws-api ; docker history kws-api:1.0.0 ; docker history kws-api:single
docker exec kws python -m src.evaluate --artifacts /app/artifacts --data-root /tmp/data
du -sh .venv
```
