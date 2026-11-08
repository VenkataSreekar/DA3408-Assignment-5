"""smoke_eta.py -- release gate for an ETA endpoint (Part D).

  python smoke_eta.py --url http://localhost:8000/v1/models/eta:predict --expect-version v1 [--max-p95-ms 80]

Exit 0 if every check passes, 1 otherwise. Works against the Part A (REST) and Part C (Ray, needs
--customer-id) servers; both return {"predictions": [...], "model_version"/"arm": ...}.
"""
import argparse
import sys
import time

import requests

BASE = dict(distance_km=20.0, n_stops=5, hour=13, weekday=2, traffic_index=0.3, rain=0, weight_kg=10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    ap.add_argument("--customer-id", default=None, help="add this customer_id to every request (Ray router)")
    ap.add_argument("--n-latency", type=int, default=50)
    a = ap.parse_args()
    sess = requests.Session()
    results = []

    def body(rows):
        b = {"instances": rows}
        if a.customer_id:
            b["customer_id"] = a.customer_id
        return b

    def predict(rows):
        r = sess.post(a.url, json=body(rows), timeout=10)
        return r

    def one(**over):
        r = predict([dict(BASE, **over)])
        r.raise_for_status()
        return r.json()["predictions"][0]

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    def c_version():
        j = predict([BASE]).json()
        got = j.get("model_version") or j.get("arm")
        return got == a.expect_version, f"got {got!r}, expected {a.expect_version!r}"

    def c_shape():
        rows = [dict(BASE, distance_km=d) for d in (5, 15, 30)]
        r = predict(rows); j = r.json()
        ok = r.status_code == 200 and len(j["predictions"]) == 3 and all(isinstance(p, (int, float)) for p in j["predictions"])
        return ok, f"status {r.status_code}, {len(j.get('predictions', []))} predictions"

    def c_sane_range():
        p = one()
        return 0 < p < 600, f"ETA for a 20 km / 5 stop job = {p} min"

    def c_distance():
        short, long_ = one(distance_km=10), one(distance_km=80)
        return long_ > short, f"10 km -> {short}, 80 km -> {long_}"

    def c_traffic():
        low, high = one(traffic_index=0.0), one(traffic_index=1.0)
        return high > low, f"traffic 0 -> {low}, traffic 1 -> {high}"

    def c_stops():
        few, many = one(n_stops=2), one(n_stops=18)
        return many > few, f"2 stops -> {few}, 18 stops -> {many}"

    def c_422():
        r = predict([dict(BASE, hour=99)])
        return r.status_code == 422, f"status {r.status_code}"

    def c_infeasible():
        r = predict([dict(BASE, distance_km=200, n_stops=30)])
        return r.status_code == 422, f"status {r.status_code}"

    def c_latency():
        lat = []
        for _ in range(a.n_latency):
            t0 = time.perf_counter(); predict([BASE]); lat.append((time.perf_counter() - t0) * 1000)
        lat.sort(); p95 = lat[int(0.95 * len(lat)) - 1]
        return p95 <= a.max_p95_ms, f"p95 = {p95:.1f} ms over {len(lat)} calls (limit {a.max_p95_ms} ms)"

    check("expected model version", c_version)
    check("batch response shape", c_shape)
    check("ETA in plausible range", c_sane_range)
    check("sanity: longer distance => longer ETA", c_distance)
    check("sanity: heavier traffic => longer ETA", c_traffic)
    check("sanity: more stops => longer ETA", c_stops)
    check("invalid input => 422", c_422)
    check("infeasible combination => 422", c_infeasible)
    check("latency p95 within limit", c_latency)

    ok = all(results)
    print(f"\n{sum(results)}/{len(results)} checks passed -> {'RELEASE ALLOWED' if ok else 'RELEASE BLOCKED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
