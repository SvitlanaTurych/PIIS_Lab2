"""Холодний старт і RAM локального запуску (без Docker).
python scripts/local_cold_start.py   (artifacts/ має існувати)
"""
import subprocess
import sys
import time
import urllib.request

import psutil

t0 = time.perf_counter()
p = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app", "--port", "8000"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while True:
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=1)
        break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(0.05)
print(f"cold start: {time.perf_counter() - t0:.2f} s")
print(f"RAM (RSS): {psutil.Process(p.pid).memory_info().rss / 1e6:.1f} MB")
p.terminate()
