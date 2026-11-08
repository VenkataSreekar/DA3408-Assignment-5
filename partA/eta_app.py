"""eta_app.py -- SwiftParcel ETA REST service (Part A).

Run:  ETA_MODEL=data/eta_v1.joblib uvicorn eta_app:app --port 8000
"""
import os
import re
import threading
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- model
# Loaded exactly once, at import time. Tests can monkeypatch `eta_app.model`.
MODEL_PATH = os.environ.get("ETA_MODEL", "data/eta_v1.joblib")
_bundle = joblib.load(MODEL_PATH)
model = _bundle["model"]
FEATURE_COLUMNS = list(_bundle["feature_columns"])
MODEL_VERSION = str(_bundle["version"])

MAX_BATCH = 500


# --------------------------------------------------------------------------- schema
class Instance(BaseModel):
    distance_km: float = Field(ge=0.1, le=500, allow_inf_nan=False)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1, allow_inf_nan=False)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000, allow_inf_nan=False)

    @model_validator(mode="after")
    def _feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError(
                "infeasible delivery: distance_km > 150 together with n_stops > 20 is not allowed"
            )
        return self


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1, max_length=MAX_BATCH)


# --------------------------------------------------------------------------- metrics
# Hand-written Prometheus text exposition (no extra dependency).
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counts = {}                       # (path, status) -> int
        self.bucket_counts = [0] * len(BUCKETS)  # non-cumulative per bucket
        self.total = 0
        self.sum = 0.0

    def observe(self, path: str, status: int, seconds: float):
        with self._lock:
            key = (path, str(status))
            self.counts[key] = self.counts.get(key, 0) + 1
            self.total += 1
            self.sum += seconds
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    self.bucket_counts[i] += 1
                    break

    def render(self) -> str:
        with self._lock:
            lines = ["# HELP eta_requests_total Requests by path and status.",
                     "# TYPE eta_requests_total counter"]
            for (path, status), n in sorted(self.counts.items()):
                lines.append(f'eta_requests_total{{path="{path}",status="{status}"}} {n}')
            lines += ["# HELP eta_request_latency_seconds Request latency.",
                      "# TYPE eta_request_latency_seconds histogram"]
            cum = 0
            for b, n in zip(BUCKETS, self.bucket_counts):
                cum += n
                lines.append(f'eta_request_latency_seconds_bucket{{le="{b}"}} {cum}')
            lines.append(f'eta_request_latency_seconds_bucket{{le="+Inf"}} {self.total}')
            lines.append(f"eta_request_latency_seconds_sum {self.sum:.6f}")
            lines.append(f"eta_request_latency_seconds_count {self.total}")
            return "\n".join(lines) + "\n"


metrics = Metrics()
KNOWN_PATHS = {"/v1/models/eta:predict", "/healthz"}

# --------------------------------------------------------------------------- app
app = FastAPI(title="SwiftParcel ETA")
_ID_OK = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Assign/echo X-Request-ID, time the request, record metrics."""
    incoming = request.headers.get("x-request-id", "")
    rid = incoming if _ID_OK.match(incoming) else str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # last line of defence: never leak internals
        response = JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    response.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":
        path = request.url.path if request.url.path in KNOWN_PATHS else "other"
        metrics.observe(path, response.status_code, time.perf_counter() - t0)
    return response


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "msg": e["msg"]}
               for e in exc.errors()]
    return JSONResponse({"error": "validation_failed", "details": details,
                         "request_id": _request_id(request)}, status_code=422)


@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    # plain `def` -> FastAPI runs it in the threadpool, so the event loop stays free
    rid = _request_id(request)
    try:
        df = pd.DataFrame([i.model_dump() for i in req.instances])[FEATURE_COLUMNS]
        preds = [round(float(p), 1) for p in model.predict(df)]
    except Exception:
        return JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    return {"predictions": preds, "model_version": MODEL_VERSION, "request_id": rid}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
def get_metrics():
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")
