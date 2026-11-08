"""eta_mlflow.py -- MLflow registry, multi-criteria promotion gate (Part B).

  export MLFLOW_TRACKING_URI=sqlite:///mlflow.db       # default if unset
  python eta_mlflow.py register
  python eta_mlflow.py gate [--min-rel-gain 0.6]
"""
import argparse
import json
import os
import time

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import eta_data

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "eta-model"
RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)
EPS = 1e-12  # absorbs float noise at exact thresholds


# ----------------------------------------------------------------------------- B1: pyfunc
class EtaPyfunc(mlflow.pyfunc.PythonModel):
    """Wraps a {model, feature_columns, version} bundle."""

    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.cols, self.version = b["model"], list(b["feature_columns"]), str(b["version"])

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.cols]
        eta = np.round(self.model.predict(X), 1)
        return pd.DataFrame({"eta_minutes": eta, "model_version": self.version})


def _log_one(bundle_path, version_label, example):
    out_example = pd.DataFrame({"eta_minutes": [0.0] * len(example), "model_version": [version_label] * len(example)})
    signature = infer_signature(example, out_example)
    reqs = [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}", f"numpy=={np.__version__}", f"joblib=={joblib.__version__}"]
    with mlflow.start_run(run_name=f"eta-{version_label}"):
        mlflow.log_param("bundle_version", version_label)
        kwargs = dict(python_model=EtaPyfunc(), artifacts={"bundle": bundle_path}, signature=signature,
                      input_example=example, pip_requirements=reqs, registered_model_name=MODEL_NAME)
        try:
            info = mlflow.pyfunc.log_model(name="model", **kwargs)          # MLflow >= 3
        except TypeError:
            info = mlflow.pyfunc.log_model(artifact_path="model", **kwargs)  # MLflow 2.x
    return info


def register(data_dir="data"):
    """Log v1 and v2, register as eta-model, set aliases prod->v1, candidate->v2. Returns the registry table."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("swiftparcel-eta")
    example = pd.read_csv(f"{data_dir}/eta_fresh.csv")[eta_data.FEATURES].head(5)
    client = MlflowClient()
    versions = {}
    for label in ("v1", "v2"):
        info = _log_one(f"{data_dir}/eta_{label}.joblib", label, example)
        mv = getattr(info, "registered_model_version", None)
        if mv is None:  # fall back: newest version registered from this run
            mv = max(int(m.version) for m in client.search_model_versions(f"name='{MODEL_NAME}'"))
        versions[label] = str(mv)
    client.set_registered_model_alias(MODEL_NAME, "prod", versions["v1"])
    client.set_registered_model_alias(MODEL_NAME, "candidate", versions["v2"])
    return registry_table()


def registry_table():
    """(version, alias, bundle label) for every registered version of eta-model."""
    client = MlflowClient(tracking_uri=TRACKING_URI)
    by_version = {}
    for alias, ver in client.get_registered_model(MODEL_NAME).aliases.items():
        by_version.setdefault(str(ver), []).append(alias)
    rows = []
    for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda m: int(m.version)):
        rows.append({"version": mv.version, "alias": ",".join(sorted(by_version.get(str(mv.version), []))) or "-",
                     "run_id": mv.run_id[:8]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- B2: gate
def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    """Pure promotion rule. prod / cand: dicts with mae, mae_rush, mae_rain, p95_ms.

    * overall MAE must improve by >= min_rel_gain relative to prod
    * candidate single-row p95 <= max_p95_ms
    * mae_rush and mae_rain must not exceed prod's by more than max_slice_regression (relative)
    """
    gain = (prod["mae"] - cand["mae"]) / prod["mae"] if prod["mae"] > 0 else (0.0 if cand["mae"] >= prod["mae"] else 1.0)
    limit = 1.0 + max_slice_regression
    criteria = {
        "overall_gain": bool(gain >= min_rel_gain - EPS),
        "latency": bool(cand["p95_ms"] <= max_p95_ms + EPS),
        "slice_rush": bool(cand["mae_rush"] <= prod["mae_rush"] * limit + EPS),
        "slice_rain": bool(cand["mae_rain"] <= prod["mae_rain"] * limit + EPS),
    }
    return {"criteria": criteria, "promote": all(criteria.values()),
            "details": {"rel_gain": round(gain, 4),
                        "rush_ratio": round(cand["mae_rush"] / prod["mae_rush"], 4) if prod["mae_rush"] else None,
                        "rain_ratio": round(cand["mae_rain"] / prod["mae_rain"], 4) if prod["mae_rain"] else None,
                        "cand_p95_ms": round(cand["p95_ms"], 3)}}


def _evaluate(model_uri, fresh, n_latency=200):
    m = mlflow.pyfunc.load_model(model_uri)
    X, y = fresh[eta_data.FEATURES], fresh["eta_minutes"].to_numpy()
    pred = m.predict(X)["eta_minutes"].to_numpy()
    err = np.abs(pred - y)
    rush = fresh["hour"].isin(RUSH_HOURS).to_numpy()
    rain = (fresh["rain"] == 1).to_numpy()
    rows = [X.iloc[[i]] for i in range(n_latency)]
    m.predict(rows[0])  # warm-up, not measured
    lat = []
    for r in rows:
        t0 = time.perf_counter(); m.predict(r); lat.append((time.perf_counter() - t0) * 1000)
    return {"mae": float(err.mean()), "mae_rush": float(err[rush].mean()), "mae_rain": float(err[rain].mean()),
            "p95_ms": float(np.percentile(lat, 95))}


def promotion_gate(**thresholds):
    """Load both aliases, evaluate on fresh data, apply decide()."""
    mlflow.set_tracking_uri(TRACKING_URI)
    fresh = eta_data.make_data(2000, seed=99)
    prod = _evaluate(f"models:/{MODEL_NAME}@prod", fresh)
    cand = _evaluate(f"models:/{MODEL_NAME}@candidate", fresh)
    out = decide(prod, cand, **thresholds)
    out.update(prod=prod, candidate=cand, thresholds=thresholds)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "table", "gate"])
    ap.add_argument("--min-rel-gain", type=float)
    ap.add_argument("--max-p95-ms", type=float)
    ap.add_argument("--max-slice-regression", type=float)
    a = ap.parse_args()
    if a.cmd == "register":
        print(register().to_string(index=False))
    elif a.cmd == "table":
        print(registry_table().to_string(index=False))
    else:
        th = {k: v for k, v in dict(min_rel_gain=a.min_rel_gain, max_p95_ms=a.max_p95_ms,
                                    max_slice_regression=a.max_slice_regression).items() if v is not None}
        print(json.dumps(promotion_gate(**th), indent=2))
