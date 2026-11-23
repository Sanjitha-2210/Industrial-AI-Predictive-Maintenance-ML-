"""Dashboard <-> backend HTTP client: error handling (mocked transport) and integration (in-process API)."""
import ast
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from src import config as C
from src.data.load_data import dataset_available

sys.path.insert(0, str(C.ROOT / "dashboard"))
from api_client import ApiClient, ApiError, api_url_from_env  # noqa: E402

need_all = pytest.mark.skipif(not (C.LSTM_CHECKPOINT.exists() and dataset_available()),
                              reason="trained model or dataset not available")


def mock_client(handler):
    return ApiClient("http://backend.test", timeout=1,
                     client=httpx.Client(base_url="http://backend.test", transport=httpx.MockTransport(handler)))


def raise_(exc):
    def handler(request):
        raise exc
    return handler


def test_backend_url_from_env(monkeypatch):
    monkeypatch.delenv("RUL_API_URL", raising=False)
    assert api_url_from_env() == "http://127.0.0.1:8000"
    monkeypatch.setenv("RUL_API_URL", "http://10.0.0.5:9000/")
    assert api_url_from_env() == "http://10.0.0.5:9000" and ApiClient().base_url == "http://10.0.0.5:9000"


def test_connection_refused_real_socket():
    c = ApiClient("http://127.0.0.1:9", timeout=2)  # nothing listens on port 9
    with pytest.raises(ApiError) as e:
        c.health()
    assert e.value.kind == "unavailable" and "Cannot reach the backend" in str(e.value) and "uvicorn" in str(e.value)


@pytest.mark.parametrize("exc,kind", [(httpx.ConnectError("refused"), "unavailable"),
                                      (httpx.ReadTimeout("slow"), "timeout"),
                                      (httpx.ConnectTimeout("slow"), "unavailable"),
                                      (httpx.WriteTimeout("slow"), "timeout"),
                                      (httpx.RemoteProtocolError("reset"), "unavailable")])
def test_transport_failures(exc, kind):
    with pytest.raises(ApiError) as e:
        mock_client(raise_(exc)).health()
    assert e.value.kind == kind


def test_http_error_detail_is_surfaced():
    c = mock_client(lambda r: httpx.Response(503, json={"detail": "Trained model not found"}))
    with pytest.raises(ApiError) as e:
        c.predict(engine_id=1)
    assert e.value.kind == "http" and e.value.status == 503 and "Trained model not found" in str(e.value)


def test_validation_error_list_is_readable():
    detail = [{"loc": ["body", "readings", 0, "sensor_2"], "msg": "Field required"}]
    with pytest.raises(ApiError) as e:
        mock_client(lambda r: httpx.Response(422, json={"detail": detail})).predict(readings=[{}])
    assert "body.readings.0.sensor_2: Field required" in str(e.value)


@pytest.mark.parametrize("response", [httpx.Response(200, text="<html>not json</html>"),
                                      httpx.Response(200, json=[1, 2]),
                                      httpx.Response(200, json={"predicted_rul": 5.0})])
def test_invalid_responses(response):
    with pytest.raises(ApiError) as e:
        mock_client(lambda r: response).predict(engine_id=1)
    assert e.value.kind == "invalid"


def test_trajectory_length_mismatch_and_unflagged_simulation():
    c = mock_client(lambda r: httpx.Response(200, json={"cycles": [1, 2], "predicted_rul": [3.0], "rul_clip": 125}))
    with pytest.raises(ApiError):
        c.trajectory(engine_id=1)
    c = mock_client(lambda r: httpx.Response(200, json={"readings": [], "label": "x", "simulated": False}))
    with pytest.raises(ApiError):
        c.simulate(0, 60)


def test_request_bodies():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.content))
        return httpx.Response(200, json={"cycles": [], "predicted_rul": [], "rul_clip": 125})
    c = mock_client(handler)
    c.trajectory(engine_id=3)
    c.trajectory(readings=[{"cycle": 1}])
    assert seen[0][:2] == ("POST", "/trajectory") and b'"engine_id":3' in seen[0][2].replace(b" ", b"")
    assert b'"readings"' in seen[1][2] and b"engine_id" not in seen[1][2]


def test_dashboard_does_not_import_model_code():
    """The dashboard must reach the model only through HTTP."""
    for f in ("app.py", "api_client.py", "table_view.py", "components.py", "theme.py"):
        tree = ast.parse((C.ROOT / "dashboard" / f).read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        bad = {m for m in mods if m.split(".")[0] in ("src", "torch", "backend", "pandas")}
        assert not bad, (f, bad)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "Predictor" not in names, f


# ---- integration: real API code, in-process transport ----
@pytest.fixture(scope="module")
def api():
    from backend.main import app
    return ApiClient("http://testserver", client=TestClient(app))


@need_all
def test_client_real_mode_flow(api):
    assert api.health()["model_loaded"]
    ids = api.engines()["engine_ids"]
    info = api.engine(ids[33])
    assert info["engine_id"] == 34 and len(info["recent_readings"]) == info["n_cycles"] == 203
    p, t = api.predict(engine_id=34), api.trajectory(engine_id=34)
    assert p["health_state"] == "Near Failure" and p["state_latched"] and p["true_rul"] == 7
    assert abs(t["predicted_rul"][-1] - p["predicted_rul"]) < 1e-2 and t["health_states"][-1] == "Near Failure"
    with pytest.raises(ApiError) as e:
        api.predict(engine_id=99999)
    assert e.value.status == 404


@need_all
def test_client_simulated_mode_flow(api):
    s = api.simulate(3, 120)
    p, t = api.predict(readings=s["readings"]), api.trajectory(readings=s["readings"])
    from src.health_state import HEALTH_STATES
    assert p["engine_id"] is None and p["health_state"] in HEALTH_STATES
    assert len(t["cycles"]) == 120


@need_all
def test_client_performance_endpoints(api):
    assert api.metrics()["test"]["n_engines"] == 100
    assert len(api.training_history()["train_loss"]) > 0 and api.test_predictions()["count"] == 100
