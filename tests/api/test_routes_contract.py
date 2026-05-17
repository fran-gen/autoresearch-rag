from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_commit_pipeline_route_is_disabled():
    response = client.post("/api/research/commit-pipeline")

    assert response.status_code == 403
    assert "disabled" in response.json()["detail"].lower()


def test_start_research_rejects_invalid_mode():
    response = client.post("/api/research/start", params={"research_mode": "invalid"})

    assert response.status_code in {400, 500}
