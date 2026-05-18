from fastapi.testclient import TestClient

from src.api.main import app
from src.api import routes


class _FakeStore:
    async def list_experiments(self):
        return []

    async def list_results(self):
        return []

    async def list_hypotheses(self):
        return []


client = TestClient(app)


def test_research_status_empty_shape():
    response = client.get("/api/research/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["running"] is False
    assert payload["latest_summary"] == "No research run has started yet."


def test_dataset_status_with_monkeypatched_readiness(monkeypatch):
    monkeypatch.setattr(
        routes,
        "_dataset_readiness",
        lambda: {
            "ready": True,
            "message": "ok",
            "documents": 1,
            "questions": "available",
        },
    )

    response = client.get("/api/dataset/status")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_store_backed_endpoints_with_fake_store():
    routes.configure_store(_FakeStore())

    experiments = client.get("/api/experiments")
    leaderboard = client.get("/api/leaderboard")
    hypotheses = client.get("/api/hypotheses")

    assert experiments.status_code == 200
    assert experiments.json() == []
    assert leaderboard.status_code == 200
    assert leaderboard.json() == []
    assert hypotheses.status_code == 200
    assert hypotheses.json() == []
