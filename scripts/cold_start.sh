#!/usr/bin/env bash
# Час від `docker run` до першої успішної відповіді /health.
# Використання: scripts/cold_start.sh kws-api:1.0.0
IMG=${1:-kws-api:1.0.0}
docker rm -f kws-cold >/dev/null 2>&1
START=$(date +%s.%N)
docker run -d --name kws-cold -p 8000:8000 "$IMG" >/dev/null
until curl -sf http://localhost:8000/health >/dev/null; do sleep 0.05; done
END=$(date +%s.%N)
echo "cold start: $(echo "$END - $START" | bc) s"
docker stats --no-stream kws-cold
docker rm -f kws-cold >/dev/null
