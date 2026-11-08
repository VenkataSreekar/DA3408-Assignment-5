"""run_a3.py -- reproduces the A3 measurements end to end on YOUR machine.

  python run_a3.py                 # baseline: 1 uvicorn worker -> loadtest_results.json
  python run_a3.py --workers 2     # the change: N uvicorn workers -> loadtest_results_w2.json

Starts the service itself, runs load_test.py at concurrency 1 2 4 8 16 (4 s each), then batch 1 vs
batch 100 at concurrency 1, and prints the tables used in REPORT.md.
Needs data/ (python eta_data.py + the eta_sample.csv command from the starter README).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import pandas as pd
import requests

import load_test

URL = "http://127.0.0.1:8000/v1/models/eta:predict"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--duration", type=float, default=4)
    a = ap.parse_args()
    env = dict(os.environ, ETA_MODEL="data/eta_v1.joblib")
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "eta_app:app", "--port", "8000",
                            "--workers", str(a.workers), "--log-level", "warning"], env=env)
    try:
        for _ in range(60):
            try:
                if requests.get("http://127.0.0.1:8000/healthz", timeout=1).ok:
                    break
            except Exception:
                time.sleep(0.5)
        rows = pd.read_csv("data/eta_sample.csv").to_dict("records")
        load_test.run(URL, "rest", 1, 1, rows, 1)             # warm-up, discarded
        sweep = [load_test.run(URL, "rest", c, a.duration, rows, 1) for c in (1, 2, 4, 8, 16)]
        b1 = load_test.run(URL, "rest", 1, a.duration, rows, 1)
        b100 = load_test.run(URL, "rest", 1, a.duration, rows, 100)
        for r, b in ((b1, 1), (b100, 100)):
            r["batch"] = b
            r["ms_per_row"] = round(1000 / (r["rps"] * b), 3)
        out = {"workers": a.workers, "cpu_count": os.cpu_count(),
               "concurrency_sweep": sweep, "batch1": b1, "batch100": b100}
        name = "loadtest_results.json" if a.workers == 1 else f"loadtest_results_w{a.workers}.json"
        json.dump(out, open(name, "w"), indent=2)
        print(f"workers={a.workers}  cpu_count={os.cpu_count()}")
        print(pd.DataFrame(sweep).drop(columns=["replicas", "versions"]).to_string(index=False))
        print(pd.DataFrame([b1, b100])[["batch", "rps", "p50_ms", "p95_ms", "ms_per_row"]].to_string(index=False))
    finally:
        srv.send_signal(signal.SIGINT)
        srv.wait(10)


if __name__ == "__main__":
    main()
