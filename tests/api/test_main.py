from fastapi.testclient import TestClient

from src.api.main import app


def test_up_endpoint_returns_ok():
    client = TestClient(app)
    response = client.get("/up")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
