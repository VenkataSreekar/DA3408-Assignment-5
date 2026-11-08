"""verify_ray.py -- demonstrates the sticky canary on a 2-CPU Ray (Part C2).
Run from partC/ :   python verify_ray.py | tee verify_output.txt
"""
import concurrent.futures as cf
import time

import ray
import requests
from ray import serve

from eta_ray import build, arm_for

PORT = 8123
URL = f"http://127.0.0.1:{PORT}/v1/models/eta:predict"
ROW = dict(distance_km=20.0, n_stops=5, hour=13, weekday=2, traffic_index=0.3, rain=0, weight_kg=10.0)
IDS = [f"cust-{i:05d}" for i in range(1000)]


def call(cid, headers=None):
    r = requests.post(URL, json={"customer_id": cid, "instances": [ROW]}, headers=headers or {}, timeout=30)
    r.raise_for_status()
    return r.json()["arm"]


def call_all(ids, headers=None):
    with cf.ThreadPoolExecutor(8) as ex:
        return dict(zip(ids, ex.map(lambda c: call(c, headers), ids)))


def deploy(weight):
    serve.run(build(weight=weight), name="eta", route_prefix="/", blocking=False)
    for _ in range(120):
        try:
            call("warmup"); return
        except Exception:
            time.sleep(1)
    raise RuntimeError("app did not come up")


def main():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    serve.start(http_options={"host": "127.0.0.1", "port": PORT})
    print(f"Ray CPUs total={ray.cluster_resources()['CPU']}")


    deploy(0.2)
    print(f"CPUs still free with the whole app deployed: {ray.available_resources().get('CPU', 0):.2f}")
    first = call_all(IDS)
    share = sum(a == "v2" for a in first.values()) / len(IDS)
    print(f"(a) candidate share over {len(IDS)} distinct customers = {share:.3f}  (target 0.200 +/- 0.040)")
    assert abs(share - 0.2) <= 0.04
    assert all(first[c] == arm_for(c, 0.2) for c in IDS), "router disagrees with arm_for()"


    sample = IDS[:200]
    second = call_all(sample)
    same = sum(first[c] == second[c] for c in sample)
    print(f"(b) {same}/200 customers got the same arm on a second call")
    assert same == 200


    serve.delete("eta"); time.sleep(2)
    deploy(0.2)
    again = call_all(IDS)
    same_all = sum(first[c] == again[c] for c in IDS)
    print(f"(c) after delete + identical redeploy: {same_all}/{len(IDS)} customers on the same arm")
    assert same_all == len(IDS)


    deploy(0.5)
    raised = call_all(IDS)
    was_cand = [c for c in IDS if first[c] == "v2"]
    moved_back = [c for c in was_cand if raised[c] != "v2"]
    share2 = sum(a == "v2" for a in raised.values()) / len(IDS)
    print(f"(d) weight 0.5: share = {share2:.3f}; customers that were on v2 = {len(was_cand)}, "
          f"moved back to v1 = {len(moved_back)}")
    assert not moved_back and abs(share2 - 0.5) <= 0.05


    prod_customers = [c for c in IDS if raised[c] == "v1"][:50]
    forced = call_all(prod_customers, headers={"X-Beta": "1"})
    plain = call_all(prod_customers)
    print(f"(e) 50 prod customers with X-Beta: 1 -> {sum(a == 'v2' for a in forced.values())}/50 served by v2; "
          f"without the header -> {sum(a == 'v1' for a in plain.values())}/50 still v1")
    assert all(a == "v2" for a in forced.values()) and all(a == "v1" for a in plain.values())

    print("\nALL CHECKS PASSED")
    serve.shutdown(); ray.shutdown()


if __name__ == "__main__":
    main()
