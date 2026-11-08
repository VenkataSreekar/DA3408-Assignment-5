# Module 4 – Assignment 5: SwiftParcel ETA service

Roll no: **DA24B030** · Name: **Venkata Sreekar**

## Machine and environment (used for every number below)
ASUS Zenbook 14 (UX3405CA), Intel Core Ultra 5 225H, 14 CPUs (`lscpu`: 14 cores, 1 thread per core, `os.cpu_count()` = 14), 14 GiB RAM
(about 7.7 GiB already in use by other applications during my runs), Linux, conda environment `aiops`, Python 3.13.15.
Packages (installed with pip into the conda env): fastapi 0.141.1, uvicorn 0.52.4, scikit-learn 1.9.0, pandas 2.3.3, mlflow 3.15.2, ray 2.59.0.
The load generator ran on the same machine as the server. Power: All runs were on battery power with the balanced power profile (not plugged in, not in performance mode).
`lscpu` reported the CPU at 22 % of its maximum frequency at an idle moment, so frequency scaling is a source of noise.

## Sources and assistance
- Starter files (`eta_data.py`, `load_test.py`, `requirements.txt`, used unmodified);
- Lecture 10 notebooks REST_API_Serving, MLflow_Serving and Ray_Serve; Lecture 9 Monitoring via Prometheus.
- Documentation: FastAPI/pydantic, MLflow (pyfunc, registry aliases, models serve), Ray Serve (deployments, autoscaling), Prometheus text format; Little's Law for the load-test analysis.
- AI assistance: I generated the .py files with AI (Claude, Sonnet 5.5, Medium).

---

## Part A – REST service

`eta_app.py` loads the bundle once at import (module-level `model`), validates with pydantic (all ranges, batch 1–500, the cross-field
"infeasible" rule → 422), puts `X-Request-ID` on every response via a middleware (client's id echoed, otherwise a UUID; same id in the JSON body),
returns the specified 422 body, renders its own Prometheus text (`eta_requests_total{path,status}` and the histogram
`eta_request_latency_seconds`; `/metrics` itself is not counted), and answers inference failures with a generic 500.
`test_eta_app.py` has 19 pytest cases and all pass on my machine: valid request, batch order, seven out-of-range cases, missing field, wrong type,
malformed body, the cross-field rule (including feasible boundary cases), both batch-size limits, request-id echo and generation (also in an error),
`/healthz`, the metrics output, and a generic 500 triggered with `monkeypatch`.
A manual check with `curl` showed the echoed `X-Request-ID`, the `/healthz` and predict counts in `/metrics`, and no `/metrics` line for itself.

### A3.1 Load test (v1, 1 uvicorn worker, 4 s per level; `python run_a3.py`)
| concurrency | requests | rps | p50 ms | p95 ms | p99 ms | errors |
|---|---|---|---|---|---|---|
| 1 | 725 | 181.2 | 5.0 | 8.4 | 11.1 | 0 |
| 2 | 993 | 248.2 | 7.1 | 13.8 | 22.9 | 0 |
| 4 | 656 | 164.0 | 24.8 | 35.8 | 41.0 | 0 |
| 8 | 772 | 193.0 | 39.9 | 66.2 | 91.1 | 0 |
| 16 | 663 | 165.8 | 92.0 | 147.1 | 256.2 | 0 |

### A3.2 Saturation and capacity under p95 ≤ 80 ms
Little's Law says concurrency L = throughput λ × mean latency W, so W ≈ L/λ. The measured p50 agrees with L/λ at every level
(p50 is slightly below the mean for a right-skewed distribution, hence slightly lower values):

| concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| L/λ (ms) | 5.5 | 8.1 | 24.4 | 41.5 | 96.5 |
| measured p50 (ms) | 5.0 | 7.1 | 24.8 | 39.9 | 92.0 |

**Saturation point: concurrency 2 (≈ 248 rps).** Up to there throughput grows with concurrency and latency barely moves (p50 5.0 → 7.1 ms). From
concurrency 4 on, throughput is flat at roughly 165–193 rps while p50 grows almost linearly with concurrency (24.8 → 39.9 → 92.0 ms): the single
Python process is the bottleneck and extra clients only wait in its queue, which is exactly what L = λW predicts when λ is fixed.
**Capacity under the SLO.** Levels 1, 2, 4 and 8 meet p95 ≤ 80 ms; level 16 does not (147.1 ms). The best measured SLO-compliant throughput is
**248 rps** (concurrency 2, p95 13.8 ms). The more conservative sustained figure is **193 rps** (concurrency 8, p95 66.2 ms), the highest concurrency that still
meets the SLO and lies on the plateau; concurrency 4 only reached 164 rps, so the 248 rps peak was not reproduced one level higher.
The dip is most likely run-to-run noise (one 4 s run per level, other applications running, CPU frequency scaling, load generator on the same machine); I did not investigate it further.

### A3.3 Cost per row, batch 1 vs batch 100 (concurrency 1, 1 worker)
| batch | rps (requests/s) | p50 ms | p95 ms | ms per row |
|---|---|---|---|---|
| 1 | 180.2 | 5.2 | 8.3 | **5.549** |
| 100 | 142.2 | 6.7 | 10.8 | **0.070** |

Batch 100 is about 79× cheaper per row. A request costs about 5.5 ms regardless of size, while 99 extra rows add only about 1.5 ms
((1000/142.2 − 1000/180.2) ms, i.e. ≈ 0.015 ms per additional row). So the cost is dominated by fixed per-request work – HTTP parsing, pydantic and
middleware, JSON encoding, building the DataFrame and the scikit-learn call overhead – which batching pays once for 100 rows.

### A3.4 The change: 2 uvicorn workers (`python run_a3.py --workers 2`)
Proposal: a single Python process is limited to roughly one core by the GIL and this service is CPU-bound, so two worker processes should about double capacity.
| concurrency | requests | rps | p50 ms | p95 ms | p99 ms | errors |
|---|---|---|---|---|---|---|
| 1 | 663 | 165.8 | 5.6 | 8.9 | 13.5 | 0 |
| 2 | 946 | 236.5 | 7.7 | 18.4 | 20.3 | 0 |
| 4 | 1873 | 468.2 | 7.6 | 14.3 | 20.8 | 0 |
| 8 | 1426 | 356.5 | 20.9 | 34.3 | 43.2 | 0 |
| 16 | 1400 | 350.0 | 41.9 | 80.4 | 100.2 | 0 |

| metric | 1 worker | 2 workers | ratio |
|---|---|---|---|
| peak rps | 248.2 (c=2) | 468.2 (c=4) | 1.89× |
| plateau rps (mean of c=8 and c=16) | 179.4 | 353.3 | 1.97× |
| sustained capacity under SLO (c=8) | 193.0 | 356.5 | 1.85× |
| p95 at c=16 | 147.1 ms | 80.4 ms | – |

The change roughly doubled capacity (1.85–1.97× depending on how it is measured), a little below 2×, which I attribute (without having isolated it) to run-to-run noise and to the
load generator competing for the same CPUs. At concurrency 1 there is no gain (181 → 166 rps, within noise), as expected:
one client keeps only one worker busy. At concurrency 16 p95 is 80.4 ms, just above the 80 ms SLO, so concurrency 8 or lower is the compliant operating point. Batch costs are unchanged
(batch 1: 5.46 ms/row, batch 100: 0.081 ms/row at concurrency 1). I did not measure more than 2 workers. Costs of the change: every worker keeps its own `/metrics` counters (no multiprocess aggregation) and its own copy of the model in memory.

### A3.5 Why p95/p99 and not the mean?
Latency is right-skewed: most requests are fast and a few are very slow, so the mean hides the tail that users and SLOs care about
(here at concurrency 16 p50 is 92 ms but p99 is 256 ms). A percentile also states a promise ("95 % of requests complete within 80 ms") that a mean cannot.

---

## Part B – MLflow

### B1 Registration (SQLite tracking store `sqlite:///mlflow.db`)
`EtaPyfunc` wraps a bundle and returns a DataFrame with `eta_minutes` (one decimal) and `model_version`. Both models are logged with a signature,
a 5-row input example and pinned `pip_requirements`, and registered as `eta-model`:

| version | alias | model (run id) |
|---|---|---|
| 1 | prod | v1, Ridge pipeline (29f9ebfb) |
| 2 | candidate | v2, gradient boosting (c6abd06d) |

### B2 Promotion gate
`decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05)` is pure and promotes only if: relative overall MAE gain ≥ `min_rel_gain`;
candidate p95 ≤ `max_p95_ms`; and rush-hour (hours 7–9 and 16–19) and `rain == 1` MAE each ≤ prod's × (1 + `max_slice_regression`) (slice limit is relative, all limits inclusive).
`promotion_gate` loads both aliases, evaluates on `make_data(2000, seed=99)` and measures p95 over 200 single-row predictions.
`test_gate.py` has 9 tests (clear win, identical models, gain / latency / slice boundaries just inside and just outside, custom thresholds, negative gain, input not mutated); all pass.

| run | MAE prod → cand | rush MAE | rain MAE | p95 prod / cand | criteria | promote |
|---|---|---|---|---|---|---|
| defaults | 11.96 → 6.10 (gain 48.9 %) | 19.68 → 8.58 (×0.436) | 11.62 → 6.29 (×0.541) | 1.81 / 1.74 ms | all four true | **true** |
| `min_rel_gain=0.60` | same | same | same | 1.81 / 1.76 ms | `overall_gain` false, others true | **false** |

The second run rejects v2 only because a 48.9 % gain is below the 60 % bar; latency and both slices still pass. Full output is in `partB/gate_output.txt`.

### B3 Shadow comparison (200 identical single-row requests to `mlflow models serve` on :5101 (prod) and :5102 (candidate))
| metric | value |
|---|---|
| (i) mean absolute difference | 9.48 min |
| (ii) requests differing by more than 10 min | 35.0 % |
| (iii) p95 latency, prod / candidate | 23.0 ms / 14.8 ms (p50 19.1 / 11.5 ms) |

For context, the mean prediction was 118.9 (prod) and 119.0 (candidate) minutes, and against the true ETA on those 200 rows the MAE was 11.42 (prod) vs 6.11 (candidate).

**Decision: I would promote v2 (through the canary in Part D, not straight to 100 %).** The models differ by 9.5 min on average and by more than 10 min on 35 % of requests, but v2 is
the more accurate model on labelled fresh data (MAE ≈ 6.1 vs ≈ 12.0) and on both slices, so the disagreement is the improvement rather than a risk, and the means agree (no systematic shift).
In-process single-row p95 is 1.7 ms for v2 against 1.8 ms for v1, far inside the 10 ms limit. The HTTP p95 values (23.0 vs 14.8 ms) should not be read as "v2 is faster": the in-process times are equal,
and the gap is most likely a measurement artefact (prod was always called first in each pair, CPU frequency scaling); I did not isolate the cause. Shadow traffic has no labels, so it shows how different v2 is, not that it is better.

**Why does moving an alias not change a running server?** An alias is only a pointer stored in the registry database. `mlflow models serve -m models:/eta-model@prod` resolves it to a concrete
version once at start-up, downloads that artifact and loads it into the process; nothing watches the registry afterwards, so the running process keeps serving the version it resolved.
**Two ways a running service can pick up the new version without dropping requests:** (1) rolling or blue-green replacement – start a new instance that resolves the alias, wait until it is ready,
shift traffic to it, then drain and stop the old one; (2) in-process hot reload – the service periodically (or on an admin call) re-resolves the alias, loads the new model beside the old one and swaps the reference
atomically, so in-flight requests finish on the old object. (Ray Serve's rolling redeploy of a deployment is an instance of the first idea.)

---

## Part C – Ray Serve sticky canary

`eta_ray.py`: `EtaModel` loads its bundle once per replica in `__init__` and predicts a batch. Prod runs 2 replicas × 0.25 CPU. The candidate uses
`autoscaling_config` (min 1, max 3, `target_ongoing_requests=2`, `upscale_delay_s=5`, `downscale_delay_s=30`) × 0.25 CPU. The router (FastAPI ingress, 0.25 CPU) takes `{"customer_id", "instances"}` and returns the model output plus `"arm"`.
Worst case 2·0.25 + 3·0.25 + 0.25 = 1.5 CPU ≤ 2; at minimum scale the app reserves 1.0 CPU (Ray reported 1.00 CPU still free). Routing: `bucket = sha256(salt + ":" + customer_id) / 2^64`;
the customer goes to the candidate iff `bucket < weight`; header `X-Beta: 1` forces the candidate. Since each customer's bucket is fixed, raising the weight can only add customers to the candidate.

`verify_ray.py` output on my machine (Ray started with `num_cpus=2`, 1000 distinct customer ids, weight 0.2), saved in `partC/verify_output.txt`:
```
Ray CPUs total=2.0
CPUs still free with the whole app deployed: 1.00
(a) candidate share over 1000 distinct customers = 0.207  (target 0.200 +/- 0.040)
(b) 200/200 customers got the same arm on a second call
(c) after delete + identical redeploy: 1000/1000 customers on the same arm
(d) weight 0.5: share = 0.488; customers that were on v2 = 207, moved back to v1 = 0
(e) 50 prod customers with X-Beta: 1 -> 50/50 served by v2; without the header -> 50/50 still v1
ALL CHECKS PASSED
```

### C3 Short answers
1. **Why not Python's `hash()`?** For strings Python salts `hash()` per process (hash randomisation, `PYTHONHASHSEED`), so the same customer id hashes to a different number in each replica,
   after each restart and after each redeploy – customers would hop between arms and the share would not even be stable. A cryptographic digest of a fixed salt plus the id (sha256 here) is identical in every process, on every machine, forever.
2. **Why stickiness matters.** (i) For the product: a customer who receives ETAs from two different models on consecutive requests sees an inconsistent promise and trusts the number less.
   (ii) For the experiment: the unit of analysis is the customer. If one customer contributes requests to both arms, the arms are contaminated, outcomes such as complaints or repeat orders cannot be attributed to one model, and the per-arm samples are not independent.
3. **Autoscaling parameters.** `target_ongoing_requests=2` keeps replicas lightly loaded: a predict takes a few milliseconds, so more than about two queued requests per replica means latency is already rising.
   `upscale_delay_s=5` reacts to a real burst but ignores a one-second blip, while `downscale_delay_s=30` avoids flapping and repeatedly paying the model-loading cost.
   `max_replicas=3` is what 0.25 CPU per replica allows inside the 2-CPU budget next to prod and the router (1.5 CPU worst case), and `min_replicas=1` keeps the canary warm for only 0.25 CPU.

---

## Part D – Release gate

`smoke_eta.py --url … --expect-version … [--max-p95-ms 80]` exits 0 on success and 1 on failure and runs 9 checks: expected model version; batch response shape; ETA in a plausible range;
**model-sanity** longer distance ⇒ longer ETA; **model-sanity** heavier traffic ⇒ longer ETA; **model-sanity** more stops ⇒ longer ETA; invalid input ⇒ 422; infeasible combination ⇒ 422;
p95 over 50 calls ≤ limit.
Run against the Part A service serving v1: all 9 checks passed (10 km → 42.1 min vs 80 km → 222.7 min; traffic 0 → 55.1 vs traffic 1 → 97.9; 2 stops → 55.0 vs 18 stops → 124.2; p95 5.8 ms), exit code 0.
With `--expect-version v2` against the same server the gate returned exit code 1.

### Abort criteria for a 10 % canary of v2 (checked every minute; any one triggers automatic rollback to 0 %)
1. **Errors:** canary 5xx rate above 1 % over a 5-minute window and at least 3× prod's rate over the same window.
2. **Latency:** canary p95 above 80 ms (the SLO) for 5 consecutive 1-minute windows, or above 1.5× prod's p95 over 10 minutes.
3. **Accuracy and sanity:** rolling 1-hour MAE on completed canary deliveries more than 10 % above prod's MAE over the same window (evaluated only once at least 500 canary deliveries have completed),
   or more than 0.5 % of canary predictions outside 1–600 minutes over 15 minutes.
4. **Stability:** 3 or more replica crashes, restarts or failed readiness probes within 10 minutes, or the smoke test failing on two consecutive runs.

---

## Not tested / limitations
- Only Linux with Python 3.13.15 was used; other operating systems and Python versions were not tried.
- Each load-test level is a single 4 s run on a laptop with other applications open and CPU frequency scaling; the load generator shares the machine with the server. Numbers vary from run to run (see the c=4 dip in A3.2).
- Only 1 and 2 uvicorn workers were measured. With several workers `/metrics` counters are per process (no multiprocess aggregation); no Prometheus scraper was run, only the text format was checked.
- The shadow-comparison latency gap (23.0 vs 14.8 ms) was not investigated; I did not check whether swapping the request order changes it.
- `mlflow models serve` was run with `--env-manager local`: the pinned requirements are recorded in the model, but a per-model virtualenv was not built.
- Autoscaling scale-up of the candidate under real load (for example with `extra_work_ms`) was not exercised; only the configuration, fractional-CPU reservation and sticky routing were verified.
  Ray's `num_cpus=2` is a logical scheduling budget on a 14-CPU machine, not a hard limit on physical cores.
- `smoke_eta.py` was run against the Part A service only (pass, and exit 1 for a wrong version). It was not run against the MLflow servers or the Ray router (`--customer-id`), nor with the server down.
- Blue-green rollout and in-process hot reload (B3) are explained, not implemented; the abort criteria in Part D are proposals and no real canary was run.
- Not type-hinting `EtaPyfunc.predict` makes MLflow print a warning about data validation; all runs worked (a signature is still logged, built from the explicit input and output examples); I did not test schema enforcement with bad input.
