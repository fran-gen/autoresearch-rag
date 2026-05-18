from __future__ import annotations

from src.config import get_settings
from src.dashboard import app as dashboard


class FakeKindeOAuth:
    def __init__(self, user_id: str):
        self.user_id = user_id

    def is_authenticated(self) -> bool:
        return True

    def get_user_info(self) -> dict[str, str]:
        return {"id": self.user_id}


def test_session_api_key_is_ignored_for_different_user(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    monkeypatch.setattr(dashboard, "kinde_oauth", FakeKindeOAuth("new-user"))

    try:
        with dashboard.app.test_request_context("/"):
            dashboard.session["runtime_google_api_key"] = "old-key"
            dashboard.session["runtime_google_api_key_user_id"] = "old-user"

            assert dashboard.get_session_runtime_google_api_key() == ""
            assert dashboard.has_google_api_key_for_session() is False
            assert "runtime_google_api_key" not in dashboard.session
            assert "runtime_google_api_key_user_id" not in dashboard.session
    finally:
        get_settings.cache_clear()


def test_api_request_forwards_key_only_for_owner(monkeypatch):
    captured_headers = []

    class Response:
        ok = True
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"status": "ok"}

    def fake_request(method, url, timeout, **kwargs):
        captured_headers.append(kwargs.get("headers", {}))
        return Response()

    monkeypatch.setattr(dashboard.requests, "request", fake_request)
    monkeypatch.setattr(dashboard, "kinde_oauth", FakeKindeOAuth("owner-user"))

    with dashboard.app.test_request_context("/"):
        dashboard.session["runtime_google_api_key"] = "owner-key"
        dashboard.session["runtime_google_api_key_user_id"] = "owner-user"

        payload, error = dashboard.api_request("GET", "/settings/api-key/status")

    assert error is None
    assert payload == {"status": "ok"}
    assert captured_headers == [{"X-Google-Api-Key": "owner-key"}]


def test_api_request_does_not_forward_key_after_account_switch(monkeypatch):
    captured_headers = []

    class Response:
        ok = True
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"status": "ok"}

    def fake_request(method, url, timeout, **kwargs):
        captured_headers.append(kwargs.get("headers", {}))
        return Response()

    monkeypatch.setattr(dashboard.requests, "request", fake_request)
    monkeypatch.setattr(dashboard, "kinde_oauth", FakeKindeOAuth("new-user"))

    with dashboard.app.test_request_context("/"):
        dashboard.session["runtime_google_api_key"] = "old-key"
        dashboard.session["runtime_google_api_key_user_id"] = "old-user"

        payload, error = dashboard.api_request("GET", "/settings/api-key/status")

    assert error is None
    assert payload == {"status": "ok"}
    assert captured_headers == [{}]
