"""shadow_compare.py -- send the same 200 single-row requests to prod (5101) and candidate (5102). (Part B3)
Run:  python shadow_compare.py | tee shadow_output.txt
"""
import time

import numpy as np
import pandas as pd
import requests

import eta_data

PROD, CAND = "http://127.0.0.1:5101/invocations", "http://127.0.0.1:5102/invocations"
rows = pd.read_csv("data/eta_fresh.csv")[eta_data.FEATURES].head(200).to_dict("records")


def hit(sess, url, row):
    t0 = time.perf_counter()
    r = sess.post(url, json={"dataframe_records": [row]}, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json()["predictions"][0]["eta_minutes"], ms


sp, sc = requests.Session(), requests.Session()
hit(sp, PROD, rows[0]); hit(sc, CAND, rows[0])           # warm-up, not counted
p, c, lp, lc = [], [], [], []
for row in rows:                                          # identical request to both, back to back
    a, ta = hit(sp, PROD, row); b, tb = hit(sc, CAND, row)
    p.append(a); c.append(b); lp.append(ta); lc.append(tb)
p, c = np.array(p), np.array(c)
diff = np.abs(p - c)
print(f"requests compared                 : {len(rows)}")
print(f"(i)   mean |prod - candidate|      : {diff.mean():.2f} min")
print(f"(ii)  requests differing by > 10 m : {(diff > 10).mean() * 100:.1f} %")
print(f"(iii) p95 latency  prod            : {np.percentile(lp, 95):.1f} ms   (p50 {np.percentile(lp, 50):.1f})")
print(f"      p95 latency  candidate       : {np.percentile(lc, 95):.1f} ms   (p50 {np.percentile(lc, 50):.1f})")
print(f"mean prediction  prod / candidate  : {p.mean():.1f} / {c.mean():.1f} min")
truth = pd.read_csv("data/eta_fresh.csv")["eta_minutes"].head(200).to_numpy()
print(f"(context) MAE vs truth prod / cand : {np.abs(p-truth).mean():.2f} / {np.abs(c-truth).mean():.2f} min")
