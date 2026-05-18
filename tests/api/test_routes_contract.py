from fastapi.testclient import TestClient

from src.api.main import app
from src.config import get_settings


client = TestClient(app)


def test_commit_pipeline_route_is_disabled():
    response = client.post("/api/research/commit-pipeline")

    assert response.status_code == 403
    assert "disabled" in response.json()["detail"].lower()


def test_start_research_rejects_invalid_mode():
    response = client.post("/api/research/start", params={"research_mode": "invalid"})

    assert response.status_code in {400, 500}


def test_api_key_status_uses_request_header(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    try:
        no_key = client.get("/api/settings/api-key/status")
        with_header = client.get(
            "/api/settings/api-key/status",
            headers={"X-Google-Api-Key": "session-key"},
        )
    finally:
        get_settings.cache_clear()

    assert no_key.status_code == 200
    assert no_key.json() == {"has_google_key": False}
    assert with_header.status_code == 200
    assert with_header.json() == {"has_google_key": True}
