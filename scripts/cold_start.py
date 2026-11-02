"""Час від docker run до першої успішної відповіді /health.
python scripts/cold_start.py kws-api:1.0.0
"""
import subprocess
import sys
import time
import urllib.request

img = sys.argv[1] if len(sys.argv) > 1 else "kws-api:1.0.0"
name = "kws-cold"
subprocess.run(["docker", "rm", "-f", name], capture_output=True)
t0 = time.perf_counter()
subprocess.run(["docker", "run", "-d", "--name", name, "-p", "8000:8000", img],
               check=True, capture_output=True)
while True:
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=1)
        break
    except Exception:
        time.sleep(0.05)
print(f"cold start: {time.perf_counter() - t0:.2f} s")
subprocess.run(["docker", "stats", "--no-stream", name])
subprocess.run(["docker", "rm", "-f", name], capture_output=True)
