# Module 4 – SwiftParcel ETA service: how to run from a clean checkout

Developed and run on Linux with Python 3.13 in a conda environment (any Python >= 3.10 should work).

## 0. Environment (conda)

    conda create -n aiops python=3.13 -y
    conda activate aiops
    pip install -r requirements.txt        # fastapi, uvicorn, scikit-learn, pandas, joblib, mlflow>=3, ray[serve]>=2.40, pytest, httpx, ...

## 1. Data and models (once per part folder you use)

Every part folder is self-contained (it ships `eta_data.py`; `load_test.py` is in `partA`). No model files, data or databases
are shipped. In each of `partA`, `partB`, `partC` run:

    python eta_data.py                      # writes data/eta_train.csv, data/eta_fresh.csv, data/eta_v1.joblib, data/eta_v2.joblib

and in `partA` additionally (needed by the load generator):

    python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

## 2. Part A – REST service (`cd partA`)

    pytest -q test_eta_app.py                                          # 19 tests
    ETA_MODEL=data/eta_v1.joblib uvicorn eta_app:app --port 8000       # serve v1 (default model is data/eta_v1.joblib)

    # load tests (stop any server on port 8000 first: run_a3.py starts its own)
    python run_a3.py                    # baseline, 1 worker  -> loadtest_results.json
    python run_a3.py --workers 2        # the change, 2 workers -> loadtest_results_w2.json

`run_a3.py` runs concurrency 1 2 4 8 16 (4 s each) and batch 1 vs batch 100 at concurrency 1, and prints the tables used in `REPORT.md`.

## 3. Part B – MLflow (`cd partB`)

    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db                      # also the default if unset
    python eta_mlflow.py register                                       # eta-model: v1 (alias prod), v2 (alias candidate)
    python eta_mlflow.py table                                          # registry table (version, alias)
    pytest -q test_gate.py                                              # 9 tests of decide()
    (echo "### default thresholds"; python eta_mlflow.py gate 2>/dev/null; echo; echo "### min_rel_gain=0.60"; python eta_mlflow.py gate --min-rel-gain 0.60 2>/dev/null) | tee gate_output.txt

Shadow comparison – three terminals, `conda activate aiops` and `export MLFLOW_TRACKING_URI=sqlite:///mlflow.db` in each, all inside `partB`:

    # terminal 2
    mlflow models serve -m "models:/eta-model@prod"      -p 5101 --env-manager local
    # terminal 3
    mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local
    # terminal 1, once both servers are up
    python shadow_compare.py | tee shadow_output.txt

Stop both servers with Ctrl+C afterwards.

## 4. Part C – Ray Serve sticky canary (`cd partC`)

    python verify_ray.py 2>&1 | tee verify_full.txt
    grep -E "^(Ray CPUs|CPUs still|\(a\)|\(b\)|\(c\)|\(d\)|\(e\)|ALL CHECKS)" verify_full.txt | tee verify_output.txt

`verify_ray.py` starts Ray with `num_cpus=2`, deploys `eta_ray.build(weight=0.2)` on port 8123, and asserts checks (a)–(e).
Ray prints a lot of log output; `verify_output.txt` keeps only the evidence lines. It takes a few minutes.

## 5. Part D – smoke test (`cd partD`, with a server running)

    # e.g. serve Part A's v1 in another terminal:  cd ../partA && ETA_MODEL=data/eta_v1.joblib uvicorn eta_app:app --port 8000
    python smoke_eta.py --url http://localhost:8000/v1/models/eta:predict --expect-version v1 --max-p95-ms 80
    echo "exit=$?"                                                       # 0 = release allowed, 1 = blocked

The Part C router also needs a customer id (`--customer-id some-id`, with `--expect-version v1` or `v2` according to that customer's arm).
That combination was not run (see REPORT.md, "Not tested").

## Layout

    REPORT.md                 all tables, written answers, machine description, "not tested" list
    README.md                 this file
    requirements.txt          the starter's requirements
    partA/  eta_app.py  test_eta_app.py  run_a3.py  eta_data.py  load_test.py  loadtest_results.json  loadtest_results_w2.json
    partB/  eta_mlflow.py  test_gate.py  shadow_compare.py  eta_data.py  gate_output.txt  shadow_output.txt
    partC/  eta_ray.py  verify_ray.py  eta_data.py  verify_output.txt
    partD/  smoke_eta.py
