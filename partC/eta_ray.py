"""eta_ray.py -- Ray Serve sticky canary for the ETA model (Part C).

  from eta_ray import build
  serve.run(build(weight=0.2), route_prefix="/")      # POST /v1/models/eta:predict
"""
import hashlib
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from ray import serve

SALT = "swiftparcel-eta-canary-v1"   # fixed forever: changing it would reshuffle every customer
PROD_CPU, CAND_CPU, ROUTER_CPU = 0.25, 0.25, 0.25
# worst case: 2*0.25 (prod) + 3*0.25 (candidate at max) + 0.25 (router) = 1.5 CPU  <= 2


def bucket(customer_id: str) -> float:
    """Stable value in [0, 1) for a customer id. sha256, NOT hash(): see report C3.1."""
    digest = hashlib.sha256(f"{SALT}:{customer_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def arm_for(customer_id: str, weight: float) -> str:
    """Candidate iff bucket < weight. Raising weight can only ADD customers to the candidate."""
    return "v2" if bucket(customer_id) < weight else "v1"


class Req(BaseModel):
    customer_id: str = Field(min_length=1, max_length=128)
    instances: list[dict] = Field(min_length=1, max_length=500)


@serve.deployment
class EtaModel:
    """Loads one bundle per replica (in __init__) and predicts a batch of rows."""

    def __init__(self, bundle_path: str, extra_work_ms: float = 0.0):
        b = joblib.load(bundle_path)
        self.model, self.cols, self.version = b["model"], list(b["feature_columns"]), str(b["version"])
        self.extra_work_ms = extra_work_ms
        self.replica = serve.get_replica_context().replica_tag

    def predict(self, rows: list) -> dict:
        if self.extra_work_ms:
            time.sleep(self.extra_work_ms / 1000.0)      # simulate a heavier model
        X = pd.DataFrame(rows)[self.cols]
        preds = [round(float(p), 1) for p in self.model.predict(X)]
        return {"predictions": preds, "model_version": self.version, "served_by": self.replica}


api = FastAPI()


@serve.deployment(ray_actor_options={"num_cpus": ROUTER_CPU})
@serve.ingress(api)
class Router:
    def __init__(self, prod, cand, weight: float):
        self.prod, self.cand, self.weight = prod, cand, weight

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: Req, x_beta: str | None = Header(default=None)):
        forced = x_beta == "1"
        arm = "v2" if forced else arm_for(req.customer_id, self.weight)
        handle = self.cand if arm == "v2" else self.prod
        out = await handle.predict.remote(req.instances)
        return {**out, "arm": arm, "forced": forced}

    @api.get("/healthz")
    async def healthz(self):
        return {"status": "ok", "weight": self.weight}


def build(weight: float = 0.1, extra_work_ms: float = 0.0, data_dir: str | None = None):
    data_dir = os.path.abspath(data_dir or os.environ.get("ETA_DATA_DIR", "data"))
    prod = EtaModel.options(
        name="EtaProd", num_replicas=2, ray_actor_options={"num_cpus": PROD_CPU},
    ).bind(os.path.join(data_dir, "eta_v1.joblib"), extra_work_ms)
    cand = EtaModel.options(
        name="EtaCandidate",
        ray_actor_options={"num_cpus": CAND_CPU},
        autoscaling_config={
            "min_replicas": 1, "max_replicas": 3,
            "target_ongoing_requests": 2,     # ~2 in-flight requests per replica before scaling up
            "upscale_delay_s": 5,             # react to a real burst, not a blip
            "downscale_delay_s": 30,          # keep capacity through short lulls
        },
    ).bind(os.path.join(data_dir, "eta_v2.joblib"), extra_work_ms)
    return Router.bind(prod, cand, float(weight))
