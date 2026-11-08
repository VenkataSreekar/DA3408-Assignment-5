"""Part A tests. Run from the folder that contains data/ :  pytest -q test_eta_app.py"""
import re
import pytest
from fastapi.testclient import TestClient

import eta_app

URL = "/v1/models/eta:predict"
ROW = dict(distance_km=10, n_stops=3, hour=8, weekday=1, traffic_index=0.5, rain=0, weight_kg=5)


@pytest.fixture()
def client():
    return TestClient(eta_app.app, raise_server_exceptions=False)


def post(client, rows, **kw):
    return client.post(URL, json={"instances": rows}, **kw)


def test_valid_request(client):
    r = post(client, [ROW])
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0] == round(body["predictions"][0], 1)
    assert body["model_version"] == eta_app.MODEL_VERSION
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_batch_order_preserved(client):
    rows = [dict(ROW, distance_km=d) for d in (5, 50, 20, 100, 1)]
    batch = post(client, rows).json()["predictions"]
    singles = [post(client, [r]).json()["predictions"][0] for r in rows]
    assert batch == singles


@pytest.mark.parametrize("bad", [
    dict(ROW, distance_km=0.0),            # below range
    dict(ROW, distance_km=501),            # above range
    dict(ROW, hour=24),                    # out of range int
    dict(ROW, rain=2),                     # rain must be 0/1
    dict(ROW, weight_kg=0),                # must be > 0
    dict(ROW, weight_kg=1000.5),           # must be <= 1000
    dict(ROW, traffic_index=1.1),
])
def test_out_of_range_rejected(client, bad):
    r = post(client, [bad])
    assert r.status_code == 422
    assert r.json()["error"] == "validation_failed"


def test_missing_field_rejected(client):
    row = dict(ROW); row.pop("weight_kg")
    r = post(client, [row])
    assert r.status_code == 422
    fields = [d["field"] for d in r.json()["details"]]
    assert any("weight_kg" in f for f in fields)


def test_wrong_type_rejected(client):
    r = post(client, [dict(ROW, n_stops="many")])
    assert r.status_code == 422


def test_malformed_body_rejected(client):
    r = client.post(URL, json={"rows": []})
    assert r.status_code == 422
    assert set(r.json()) == {"error", "details", "request_id"}


def test_cross_field_infeasible(client):
    r = post(client, [dict(ROW, distance_km=151, n_stops=21)])
    assert r.status_code == 422
    assert "infeasible" in r.text
    assert post(client, [dict(ROW, distance_km=150, n_stops=21)]).status_code == 200
    assert post(client, [dict(ROW, distance_km=151, n_stops=20)]).status_code == 200


def test_batch_size_limits(client):
    assert post(client, []).status_code == 422
    assert post(client, [ROW] * 500).status_code == 200
    assert post(client, [ROW] * 501).status_code == 422


def test_request_id_echo_and_generation(client):
    r = post(client, [ROW], headers={"X-Request-ID": "client-123"})
    assert r.headers["X-Request-ID"] == "client-123" and r.json()["request_id"] == "client-123"
    r = post(client, [ROW])
    assert re.fullmatch(r"[0-9a-f-]{36}", r.headers["X-Request-ID"])
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


def test_request_id_in_error_response(client):
    r = post(client, [dict(ROW, hour=99)], headers={"X-Request-ID": "err-7"})
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "err-7" and r.json()["request_id"] == "err-7"
    r = post(client, [dict(ROW, hour=99)])
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": eta_app.MODEL_VERSION}
    assert "X-Request-ID" in r.headers


def _count(text, path, status):
    m = re.search(rf'eta_requests_total\{{path="{re.escape(path)}",status="{status}"\}} (\d+)', text)
    return int(m.group(1)) if m else 0


def test_metrics_output(client):
    before = client.get("/metrics").text
    n200, n422 = _count(before, URL, 200), _count(before, URL, 422)
    post(client, [ROW]); post(client, [dict(ROW, hour=99)])
    after = client.get("/metrics").text
    assert _count(after, URL, 200) == n200 + 1
    assert _count(after, URL, 422) == n422 + 1
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in after
    assert len(re.findall(r"eta_request_latency_seconds_bucket\{le=", after)) >= 4
    assert "eta_request_latency_seconds_sum" in after and "eta_request_latency_seconds_count" in after
    assert 'path="/metrics"' not in after

def test_generic_500_hides_internals(client, monkeypatch):
    class Boom:
        def predict(self, X):
            raise RuntimeError("secret internal detail /srv/model.pkl")
    monkeypatch.setattr(eta_app, "model", Boom())
    r = post(client, [ROW])
    assert r.status_code == 500
    assert "secret" not in r.text and "RuntimeError" not in r.text and "Traceback" not in r.text
    assert "X-Request-ID" in r.headers
