import os

from fastapi.testclient import TestClient

from dashboard.server import create_app


def test_localhost_dashboard_requests_are_allowed_without_api_key(monkeypatch):
    monkeypatch.setenv("PBT_API_KEY", "secret")

    app = create_app(state_provider=lambda: {"equity": 12345.0, "generation": 2})
    client = TestClient(app, base_url="http://127.0.0.1:8000")

    response = client.get("/api/state")
    assert response.status_code == 200
    payload = response.json()
    assert payload["equity"] == 12345.0

    with client.websocket_connect("/ws") as websocket:
        message = websocket.receive_json()
        assert message["equity"] == 12345.0
