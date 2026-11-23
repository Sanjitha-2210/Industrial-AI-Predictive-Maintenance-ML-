"""HTTP client the dashboard uses to talk to the FastAPI backend (the dashboard runs no model code).

Backend URL: environment variable RUL_API_URL (default http://127.0.0.1:8000).
Timeout (seconds): environment variable RUL_API_TIMEOUT (default 30).
Every failure (connection refused, timeout, HTTP error, malformed response) is raised as ApiError
with a message that can be shown to the user as-is.
"""
from __future__ import annotations

import os

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 30.0


def api_url_from_env() -> str:
    return os.environ.get("RUL_API_URL", DEFAULT_API_URL).strip().rstrip("/") or DEFAULT_API_URL


def timeout_from_env() -> float:
    try:
        return float(os.environ.get("RUL_API_TIMEOUT", DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT


class ApiError(RuntimeError):
    """kind: 'unavailable' | 'timeout' | 'http' | 'invalid'; status: HTTP status code for kind 'http'."""

    def __init__(self, message: str, kind: str, status: int | None = None):
        super().__init__(message)
        self.kind, self.status = kind, status


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None, client: httpx.Client | None = None):
        self.base_url = (base_url or api_url_from_env()).rstrip("/")
        self.timeout = timeout if timeout is not None else timeout_from_env()
        # `client` lets tests inject an in-process transport (e.g. FastAPI's TestClient)
        self._client = client or httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _request(self, method: str, path: str, required: tuple[str, ...] = (), **kwargs) -> dict:
        try:
            r = self._client.request(method, path, **kwargs)
        except httpx.ConnectTimeout:  # could not even connect (Windows retries refused connections for ~2 s)
            raise ApiError(_unreachable(self.base_url, "connection timed out"), "unavailable")
        except httpx.TimeoutException:
            raise ApiError(f"The backend at {self.base_url} did not respond within {self.timeout:.0f} s "
                           f"({method} {path}).", "timeout")
        except httpx.TransportError as e:
            raise ApiError(_unreachable(self.base_url, type(e).__name__), "unavailable")
        if r.status_code >= 400:
            raise ApiError(f"{method} {path} failed with HTTP {r.status_code}: {_detail(r)}", "http", r.status_code)
        try:
            body = r.json()
        except ValueError:
            raise ApiError(f"{method} {path} returned a non-JSON response.", "invalid")
        if not isinstance(body, dict):
            raise ApiError(f"{method} {path} returned unexpected JSON (expected an object).", "invalid")
        missing = [k for k in required if k not in body]
        if missing:
            raise ApiError(f"{method} {path} response is missing fields: {missing}.", "invalid")
        return body

    # ---- endpoints ----
    def health(self) -> dict:
        return self._request("GET", "/health", ("model_loaded", "dataset_available", "thresholds"))

    def engines(self) -> dict:
        return self._request("GET", "/engines", ("engine_ids", "count"))

    def engine(self, engine_id: int, last_n: int = 100_000) -> dict:
        return self._request("GET", f"/engines/{int(engine_id)}", ("recent_readings", "true_rul_at_last_cycle"),
                             params={"last_n": last_n})

    def predict(self, engine_id: int | None = None, readings: list[dict] | None = None) -> dict:
        return self._request("POST", "/predict", ("predicted_rul", "health_status", "health_state", "uncertainty",
                                                  "explanation"),
                             json=_body(engine_id, readings))

    def trajectory(self, engine_id: int | None = None, readings: list[dict] | None = None) -> dict:
        body = self._request("POST", "/trajectory", ("cycles", "predicted_rul", "rul_clip"),
                             json=_body(engine_id, readings))
        if len(body["cycles"]) != len(body["predicted_rul"]):
            raise ApiError("POST /trajectory returned cycles and predictions of different lengths.", "invalid")
        return body

    def simulate(self, seed: int, n_cycles: int, stage: str = "normal") -> dict:
        body = self._request("GET", "/simulate", ("readings", "label", "stage", "scenario_rul"),
                             params={"seed": seed, "n_cycles": n_cycles, "stage": stage})
        if not body.get("simulated"):
            raise ApiError("GET /simulate response is not flagged as simulated.", "invalid")
        return body

    def metrics(self) -> dict:
        return self._request("GET", "/metrics", ("test", "validation"))

    def training_history(self) -> dict:
        return self._request("GET", "/metrics/history", ("train_loss", "val_loss"))

    def test_predictions(self) -> dict:
        return self._request("GET", "/metrics/test-predictions", ("rows",))


START_BACKEND_CMD = r".\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000"


def _unreachable(base_url: str, reason: str) -> str:
    # commands in backticks: the dashboard renders this as Markdown, which would otherwise eat backslashes
    return (f"Cannot reach the backend at {base_url} ({reason}). Start it in a second terminal from the project "
            f"folder with `{START_BACKEND_CMD}` (or start everything with `.\\start_demo.ps1`), "
            f"or set RUL_API_URL to the correct address.")


def _body(engine_id, readings) -> dict:
    return {"engine_id": int(engine_id)} if engine_id is not None else {"readings": readings}


def _detail(r: httpx.Response) -> str:
    try:
        d = r.json().get("detail", r.text)
    except (ValueError, AttributeError):
        return r.text[:300] or r.reason_phrase
    if isinstance(d, list):  # FastAPI validation errors
        return "; ".join(f"{'.'.join(map(str, e.get('loc', [])))}: {e.get('msg')}" for e in d if isinstance(e, dict))
    return str(d)
