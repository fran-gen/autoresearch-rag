import asyncio
import base64
import difflib
import json
import logging
import os
import re
import sqlite3
import sys
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from flask import Flask, Response, flash, has_request_context, jsonify, redirect, render_template, request, session, url_for
from flask_session import Session
from markupsafe import Markup, escape
import requests
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

import kinde_flask
from kinde_sdk.auth.oauth import OAuth

from src.benchmark.loader import EnterpriseRagBenchLoader
from src.config import get_settings
from src.agents.text_utils import extract_text_content

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_template_dir() -> Path:
    env_template_dir = os.getenv("FLASK_TEMPLATE_DIR")
    if env_template_dir:
        path = Path(env_template_dir).resolve()
        if path.exists():
            logger.info("Using template dir from FLASK_TEMPLATE_DIR: %s", path)
            return path

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "templates",
        here.parent.parent / "templates",
        here.parents[2] / "templates",
        here.parents[2] / "src" / "templates",
    ]

    for path in candidates:
        if path.exists():
            logger.info("Using template dir: %s", path)
            return path

    raise FileNotFoundError(
        "Could not find templates directory. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def resolve_static_dir() -> Path:
    env_static_dir = os.getenv("FLASK_STATIC_DIR")
    if env_static_dir:
        path = Path(env_static_dir).resolve()
        if path.exists():
            logger.info("Using static dir from FLASK_STATIC_DIR: %s", path)
            return path

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "static",
        here.parent.parent / "static",
        here.parents[2] / "static",
        here.parents[2] / "src" / "static",
    ]

    for path in candidates:
        if path.exists():
            logger.info("Using static dir: %s", path)
            return path

    raise FileNotFoundError(
        "Could not find static directory. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def resolve_session_file_dir() -> Path:
    env_session_dir = os.getenv("SESSION_FILE_DIR")
    if env_session_dir:
        return Path(env_session_dir).expanduser().resolve()

    return resolve_project_root() / ".flask_sessions"


BASE_DIR = resolve_project_root()
load_dotenv(BASE_DIR / ".env")


def configure_kinde_environment() -> None:
    env_aliases = {
        "KINDE_REDIRECT_URI": os.getenv("KINDE_CALLBACK_URL"),
        "KINDE_HOST": os.getenv("KINDE_ISSUER_URL"),
        "SECRET_KEY": os.getenv("FLASK_SECRET_KEY"),
    }

    for target, value in env_aliases.items():
        if value and not os.getenv(target):
            os.environ[target] = value


configure_kinde_environment()

API_BASE = os.getenv("API_BASE", "http://localhost:8001").rstrip("/")
DEFAULT_API_UPSTREAM = "http://localhost:8001"
DEFAULT_REFRESH_SECONDS = 5
MAX_RESEARCH_ITERATIONS = 10
PUBLIC_ENDPOINTS = {"up", "index", "login", "logout", "register", "callback", "dashboard_logout", "static", "api_proxy"}
AUTH_ENDPOINTS = {"login", "register", "callback", "logout"}
BACKGROUND_ENDPOINTS = {"dashboard_status_snapshot"}
_kinde_callback_url = (os.getenv("KINDE_CALLBACK_URL") or "").strip()
_parsed_callback = urlparse(_kinde_callback_url) if _kinde_callback_url else None
CANONICAL_AUTH_ORIGIN = (
    f"{_parsed_callback.scheme}://{_parsed_callback.netloc}"
    if _parsed_callback and _parsed_callback.scheme and _parsed_callback.netloc
    else None
)
NAV_ITEMS = [
    {"endpoint": "index", "label": "Overview", "icon": "⌘"},
    {"endpoint": "experiments", "label": "Experiments", "icon": "↗"},
    {"endpoint": "hypotheses", "label": "Hypotheses", "icon": "◇"},
    {"endpoint": "leaderboards", "label": "Leaderboards", "icon": "★"},
    {"endpoint": "experiment_detail_search", "label": "Experiment Detail", "icon": "◎"},
]
NAV_ACTIVE_ALIASES = {
    "experiments": {"experiments", "new_experiment"},
}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


def is_absolute_url(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


API_ROUTE_PREFIX = "/api"
API_PROXY_PREFIX = "/api"
API_UPSTREAM = (
    os.getenv("API_UPSTREAM")
    or os.getenv("INTERNAL_API_BASE")
    or (API_BASE if is_absolute_url(API_BASE) else DEFAULT_API_UPSTREAM)
).rstrip("/")


def log_dashboard_startup() -> None:
    settings = get_settings()
    loader = EnterpriseRagBenchLoader(settings.benchmark_root)
    questions_path = loader.resolved_questions_path()

    logger.info("Dashboard startup cwd: %s", Path.cwd())
    logger.info("Dashboard startup project root: %s", BASE_DIR)
    logger.info("Dashboard startup API_BASE: %s", API_BASE)
    logger.info("Dashboard startup API_UPSTREAM: %s", API_UPSTREAM)
    logger.info("Dashboard startup API_ROUTE_PREFIX: %s", API_ROUTE_PREFIX or "/")
    logger.info("Dashboard startup BENCHMARK_ROOT env: %s", os.environ.get("BENCHMARK_ROOT", "<unset>"))
    logger.info("Dashboard startup benchmark root: %s", settings.benchmark_root.resolve())
    logger.info("Dashboard startup documents dir: %s", loader.documents_dir.resolve())
    logger.info("Dashboard startup documents dir exists: %s", loader.documents_dir.exists())
    logger.info("Dashboard startup bench dir: %s", loader.bench_dir.resolve())
    logger.info("Dashboard startup questions path: %s", questions_path.resolve() if questions_path else "<missing>")
    logger.info("Dashboard startup QDRANT_URL env: %s", os.environ.get("QDRANT_URL", "<unset>"))
    logger.info("Dashboard startup qdrant url: %s", settings.qdrant_url or "<unset>")
    logger.info("Dashboard startup qdrant path fallback: %s", settings.qdrant_path.resolve())
    logger.info("Dashboard startup host repo is expected to be mounted at /app by docker-compose.yml")
    logger.info("Dashboard data status page source: API GET /api/dataset/status")
    logger.info("Dashboard host download command, full dataset: python scripts/download_dataset.py full")
    logger.info("Dashboard host download command, half dataset: python scripts/download_dataset.py half")
    logger.info("Dashboard in-container index command: docker compose exec api python scripts/embed_dataset.py")

app = Flask(
    __name__,
    template_folder=str(resolve_template_dir()),
    static_folder=str(resolve_static_dir()),
    static_url_path="/static",
)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

_secret = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")
app.config["SECRET_KEY"] = _secret
os.environ.setdefault("SECRET_KEY", _secret)

kinde_oauth = OAuth(framework="flask", app=app)

app.config["SECRET_KEY"] = _secret
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = str(resolve_session_file_dir())
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
Session(app)


def run_with_temporary_event_loop(coro):
    previous_loop = None
    try:
        previous_loop = asyncio.get_event_loop()
    except RuntimeError:
        previous_loop = None

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        if previous_loop is not None and not previous_loop.is_closed():
            asyncio.set_event_loop(previous_loop)
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())


def run_with_active_event_loop(coro: Awaitable[Any]) -> Any:
    return ensure_active_event_loop().run_until_complete(coro)


def redirect_to_async_url(coro: Awaitable[str]) -> Response:
    return redirect(run_with_active_event_loop(coro))


def get_post_login_redirect_url() -> str:
    post_login_redirect = session.pop("post_login_redirect_url", None)
    if post_login_redirect:
        target = post_login_redirect.get("url", "/")
        parsed = urlparse(target)
        if parsed.path == url_for("dashboard_status_snapshot"):
            return "/"
        return target
    return "/"


def build_reauth_login_url(reauth_state: str) -> str:
    decoded_auth_state = base64.b64decode(reauth_state).decode("utf-8")
    reauth_dict = json.loads(decoded_auth_state)

    redirect_url = os.getenv("KINDE_REDIRECT_URI")
    base_url = redirect_url.replace("/callback", "")
    login_route_url = f"{base_url}/login"

    parsed = urlparse(login_route_url)
    query_dict = parse_qs(parsed.query)
    for key, value in reauth_dict.items():
        query_dict[key] = [value]

    new_query = urlencode(query_dict, doseq=True)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
        )
    )


def install_kinde_route_override(endpoint: str, view: Callable[..., Response | dict[str, Any] | str]):
    app.view_functions[endpoint] = view


def kinde_login():
    return redirect_to_async_url(kinde_oauth.login())


def kinde_register():
    return redirect_to_async_url(kinde_oauth.register())


def kinde_logout():
    user_id = session.get("user_id")
    session.clear()
    return redirect_to_async_url(kinde_oauth.logout(user_id))


def kinde_callback():
    error = request.args.get("error")
    if error and error.lower() == "login_link_expired":
        reauth_state = request.args.get("reauth_state")
        if reauth_state:
            try:
                return redirect(build_reauth_login_url(reauth_state))
            except Exception as exc:
                return f"Error parsing reauth state: {exc}", 400

    post_login_redirect = get_post_login_redirect_url()

    code = request.args.get("code")
    state = request.args.get("state")

    if not code:
        return "Authentication failed: Missing authorization code", 400

    user_id = session.get("user_id", str(uuid.uuid4()))
    session["user_id"] = user_id

    try:
        run_with_temporary_event_loop(kinde_oauth.handle_redirect(code, user_id, state))
    except Exception as exc:
        return f"Authentication failed: {exc}", 400

    if not post_login_redirect.startswith("http"):
        post_login_redirect = str(request.url_root).rstrip("/") + post_login_redirect

    return redirect(post_login_redirect)


def kinde_user():
    try:
        if not kinde_oauth.is_authenticated():
            return redirect_to_async_url(kinde_oauth.login())
        return kinde_oauth.get_user_info()
    except Exception as exc:
        return f"Failed to get user info: {exc}", 400


install_kinde_route_override("login", kinde_login)
install_kinde_route_override("register", kinde_register)
install_kinde_route_override("logout", kinde_logout)
install_kinde_route_override("callback", kinde_callback)
install_kinde_route_override("get_user", kinde_user)


def get_authorized_data():
    if not kinde_oauth.is_authenticated():
        return {}

    try:
        user = kinde_oauth.get_user_info()
    except Exception as exc:
        logger.warning("Failed to load Kinde user info: %s", exc)
        for key in ("user_id", "live_mode", "refresh_every", "research_pending"):
            session.pop(key, None)
        return {}

    if not user:
        return {}

    return {
        "id": user.get("id"),
        "user_given_name": user.get("given_name"),
        "user_family_name": user.get("family_name"),
        "user_email": user.get("email"),
        "user_picture": user.get("picture"),
    }


def get_authenticated_user_id() -> str:
    if not kinde_oauth.is_authenticated():
        return ""

    try:
        user = kinde_oauth.get_user_info()
    except Exception as exc:
        logger.warning("Failed to load Kinde user id: %s", exc)
        return ""

    return str((user or {}).get("id") or "").strip()


def clear_runtime_google_api_key() -> None:
    session.pop("runtime_google_api_key", None)
    session.pop("runtime_google_api_key_user_id", None)


def get_session_runtime_google_api_key() -> str:
    api_key = str(session.get("runtime_google_api_key") or "").strip()
    if not api_key:
        return ""

    owner_user_id = str(session.get("runtime_google_api_key_user_id") or "").strip()
    if not owner_user_id:
        clear_runtime_google_api_key()
        return ""

    current_user_id = get_authenticated_user_id()
    if not current_user_id:
        return ""
    if owner_user_id != current_user_id:
        clear_runtime_google_api_key()
        return ""

    return api_key


def get_user_initials(user_data: dict[str, str | None]) -> str:
    first = (user_data.get("user_given_name") or "")[:1]
    last = (user_data.get("user_family_name") or "")[:1]
    initials = f"{first}{last}".strip()
    return initials.upper() or "KU"


def get_user_display_name(user_data: dict[str, str | None]) -> str:
    name_parts = [
        part.strip()
        for part in (user_data.get("user_given_name"), user_data.get("user_family_name"))
        if part and part.strip()
    ]
    return " ".join(name_parts) or user_data.get("user_email") or "User"


def build_nav_items() -> list[dict[str, str | bool]]:
    current_endpoint = request.endpoint or ""
    return [
        {
            **item,
            "href": url_for(item["endpoint"]),
            "active": current_endpoint in NAV_ACTIVE_ALIASES.get(item["endpoint"], {item["endpoint"]}),
        }
        for item in NAV_ITEMS
    ]


def current_redirect_target() -> str:
    target = request.full_path or request.path
    if target.endswith("?"):
        return target[:-1]
    return target


def remember_post_login_redirect() -> None:
    if request.endpoint in BACKGROUND_ENDPOINTS or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return
    session["post_login_redirect_url"] = {"url": current_redirect_target()}


def unauthenticated_background_response():
    """Keep long-running dashboard polling quiet when auth expires mid-run."""
    if request.endpoint == "dashboard_status_snapshot":
        return jsonify(
            {
                "should_poll": False,
                "auth_required": True,
                "sidebar_status": {
                    "reachable": False,
                    "state": "Signed out",
                    "state_class": "warning",
                    "phase": "authentication required",
                    "progress_pct": 0,
                    "message": "Sign in again to resume live dashboard updates.",
                    "last_error": None,
                    "accepted": 0,
                    "rejected": 0,
                    "best_score": None,
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                },
            }
        )
    if request.endpoint == "experiments" and request.args.get("partial") == "1":
        return Response(status=204)
    return jsonify({"detail": "Authentication required.", "auth_required": True}), 401


def ensure_active_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop


def normalize_refresh_seconds(value: int | None) -> int:
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            value = None
    if value is None:
        return DEFAULT_REFRESH_SECONDS
    return max(3, min(30, value))


def normalize_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_dashboard_api_base() -> str:
    if API_UPSTREAM.endswith(API_ROUTE_PREFIX):
        return API_UPSTREAM
    return f"{API_UPSTREAM}{API_ROUTE_PREFIX}"


def build_api_proxy_target(path: str = "") -> str:
    target = get_dashboard_api_base()
    if path:
        target = f"{target}/{path.lstrip('/')}"

    query_string = request.query_string.decode("utf-8")
    if query_string:
        target = f"{target}?{query_string}"

    return target


def build_api_proxy_headers() -> dict[str, str]:
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS and name.lower() != "host"
    }

    remote_addr = request.headers.get("X-Forwarded-For") or request.remote_addr
    if remote_addr:
        headers["X-Forwarded-For"] = remote_addr
    headers["X-Forwarded-Host"] = request.host
    headers["X-Forwarded-Proto"] = request.scheme
    return headers


def rewrite_proxy_response_headers(response_headers: requests.structures.CaseInsensitiveDict) -> list[tuple[str, str]]:
    rewritten_headers = []
    upstream_base = get_dashboard_api_base().rstrip("/")
    upstream_origin = urlparse(upstream_base)
    upstream_origin_url = f"{upstream_origin.scheme}://{upstream_origin.netloc}"
    public_origin = request.host_url.rstrip("/")

    for name, value in response_headers.items():
        if name.lower() in HOP_BY_HOP_HEADERS:
            continue

        if name.lower() == "location" and value.startswith(upstream_origin_url):
            if value.startswith(upstream_base):
                suffix = value[len(upstream_base):]
            else:
                suffix = value[len(upstream_origin_url):]
            value = f"{public_origin}{API_PROXY_PREFIX}{suffix}"

        rewritten_headers.append((name, value))

    return rewritten_headers


@app.route(API_PROXY_PREFIX, defaults={"path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"], strict_slashes=False)
@app.route(f"{API_PROXY_PREFIX}/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
def api_proxy(path: str):
    target_url = build_api_proxy_target(path)

    try:
        upstream_response = requests.request(
            method=request.method,
            url=target_url,
            headers=build_api_proxy_headers(),
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=30,
        )
    except requests.RequestException as exc:
        logger.warning("API proxy request failed for %s: %s", target_url, exc)
        return jsonify({"detail": f"Could not reach internal API at {API_UPSTREAM}."}), 502

    return Response(
        upstream_response.content,
        status=upstream_response.status_code,
        headers=rewrite_proxy_response_headers(upstream_response.headers),
    )


def get_dashboard_preferences() -> tuple[bool, int]:
    live_arg = request.args.get("live")
    refresh_arg = request.args.get("refresh")

    if live_arg is None:
        live_mode = normalize_bool(session.get("live_mode"), default=False)
    else:
        live_mode = normalize_bool(live_arg)

    refresh_every = normalize_refresh_seconds(
        refresh_arg if refresh_arg is not None else session.get("refresh_every")
    )

    session["live_mode"] = live_mode
    session["refresh_every"] = refresh_every
    return live_mode, refresh_every


def to_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    candidate = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def format_datetime(value: Any) -> str:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return "—"
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def serialize_for_template(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, indent=2, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


def normalize_display_text(value: Any, fallback: str = "") -> str:
    text = extract_text_content(value).strip()
    if not text:
        return fallback
    return text.replace("\\n", "\n")


def render_markdown(value: Any) -> Markup:
    text = normalize_display_text(value)
    if not text:
        return Markup("")

    html: list[str] = []
    paragraph: list[str] = []
    list_type: str | None = None
    in_code = False
    code_lines: list[str] = []
    code_lang = ""

    def flush_paragraph() -> None:
        if paragraph:
            html.append(f"<p>{_render_inline_markdown(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            html.append(f"</{list_type}>")
            list_type = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                html.append(
                    f"<pre><code class=\"language-{escape(code_lang)}\">"
                    f"{escape(chr(10).join(code_lines))}</code></pre>"
                )
                code_lines = []
                code_lang = ""
                in_code = False
            else:
                flush_paragraph()
                close_list()
                code_lang = stripped.strip("`").strip()[:24]
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            close_list()
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = min(4, len(heading.group(1)) + 2)
            html.append(f"<h{level}>{_render_inline_markdown(heading.group(2))}</h{level}>")
            continue

        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if bullet or ordered:
            flush_paragraph()
            target_list = "ol" if ordered else "ul"
            if list_type != target_list:
                close_list()
                html.append(f"<{target_list}>")
                list_type = target_list
            item = bullet.group(1) if bullet else ordered.group(1)
            html.append(f"<li>{_render_inline_markdown(item)}</li>")
            continue

        close_list()
        paragraph.append(stripped)

    if in_code:
        html.append(f"<pre><code>{escape(chr(10).join(code_lines))}</code></pre>")
    flush_paragraph()
    close_list()
    return Markup("\n".join(html))


def _render_inline_markdown(value: str) -> Markup:
    html = str(escape(value))
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
    html = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html)
    return Markup(html)


RESEARCH_SETUP_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_setups (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

UPLOAD_FILE_SIZE_LIMIT_BYTES = 5 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".jsonl", ".json"}


def init_research_setup_store() -> None:
    db_path = get_settings().experiment_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(RESEARCH_SETUP_SCHEMA_SQL)
        conn.commit()


def uploaded_file_size(file_storage) -> int:
    stream = file_storage.stream
    current_pos = stream.tell()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(current_pos)
    return size


def validate_uploaded_dataset_file(file_storage, label: str) -> str | None:
    filename = (file_storage.filename or "").strip()
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        return f"{label} must be a .jsonl or .json file."

    size = uploaded_file_size(file_storage)
    if size <= 0:
        return f"{label} is empty."
    if size > UPLOAD_FILE_SIZE_LIMIT_BYTES:
        max_mb = UPLOAD_FILE_SIZE_LIMIT_BYTES // (1024 * 1024)
        return f"{label} exceeds the demo upload limit of {max_mb} MB."

    return None


def build_starting_config_for_focus(question_focus: str) -> dict[str, object]:
    settings = get_settings()
    focus = (question_focus or "all").strip()
    config: dict[str, object] = {
        "strategy": "dense",
        "embedding_model": settings.embedding_model,
        "top_k": settings.default_top_k,
        "use_reranker": False,
        "reranker_model": None,
        "bm25_weight": 0.5,
        "dense_weight": 0.5,
        "evaluation_mode": "fast",
        "extra": {"question_focus": focus},
    }
    if focus in {"semantic", "multi_hop", "comparison", "analytical"}:
        config.update({"strategy": "hybrid", "top_k": 10, "use_reranker": True, "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"})
        config["extra"] = {**config["extra"], "query_rewrite": True, "source_diversity": True, "question_type_overrides": {focus: {"top_k": 12}}}
    elif focus in {"basic", "lookup", "factoid"}:
        config.update({"strategy": "dense", "top_k": 6})
        config["extra"] = {**config["extra"], "query_rewrite": False, "source_diversity": False, "question_type_overrides": {focus: {"top_k": 6}}}
    elif focus != "all":
        config.update({"strategy": "hybrid", "top_k": 8})
        config["extra"] = {**config["extra"], "query_rewrite": True, "source_diversity": True, "question_type_overrides": {focus: {"top_k": 10}}}
    else:
        config["extra"] = {**config["extra"], "query_rewrite": False, "source_diversity": True}
    return config


def describe_question_focus(question_focus: str) -> str:
    focus = (question_focus or "all").strip()
    descriptions = {
        "all": "Balanced mode across all downloaded benchmark questions. Good for broad improvements before specializing.",
        "semantic": "Optimizes for meaning-based questions where relevant wording may not exactly match the query.",
        "multi_hop": "Optimizes for questions that need evidence from more than one document or fact chain.",
        "comparison": "Optimizes for side-by-side questions that compare policies, accounts, or document details.",
        "analytical": "Optimizes for questions requiring synthesis, reasoning, or patterns across retrieved context.",
        "basic": "Optimizes for direct fact lookup with tighter retrieval and less context noise.",
        "lookup": "Optimizes for precise document lookup where exact facts or IDs matter most.",
        "factoid": "Optimizes for short factual answers where precision is more important than broad recall.",
    }
    return descriptions.get(
        focus,
        f"Specializes the starting configuration for {focus.replace('_', ' ')} questions using the downloaded benchmark labels.",
    )


def get_question_type_options() -> list[dict[str, object]]:
    settings = get_settings()
    loader = EnterpriseRagBenchLoader(settings.benchmark_root)
    questions = loader.load_questions()
    counts: dict[str, int] = {}
    previews: dict[str, list[dict[str, object]]] = {"all": []}
    for question in questions:
        qtype = question.question_type or "unknown"
        counts[qtype] = counts.get(qtype, 0) + 1
        preview = {
            "question_id": question.question_id,
            "question": question.question,
            "source_types": question.source_types,
            "expected_doc_count": len(question.expected_doc_ids),
        }
        if len(previews["all"]) < 5:
            previews["all"].append(preview)
        previews.setdefault(qtype, [])
        if len(previews[qtype]) < 5:
            previews[qtype].append(preview)
    options = [{"value": "all", "label": "Generic mix", "count": sum(counts.values()), "description": "Balanced benchmark coverage across question types.", "tooltip": describe_question_focus("all"), "starting_config": build_starting_config_for_focus("all"), "preview_questions": previews["all"]}]
    for qtype, count in sorted(counts.items(), key=lambda item: item[0]):
        options.append({"value": qtype, "label": qtype.replace("_", " ").title(), "count": count, "description": f"Tune starting config for {qtype.replace('_', ' ')} questions.", "tooltip": describe_question_focus(qtype), "starting_config": build_starting_config_for_focus(qtype), "preview_questions": previews.get(qtype, [])})
    return options


def save_uploaded_research_dataset(setup_id: str, require_upload: bool = False) -> tuple[str | None, str | None]:
    dataset_source = (request.form.get("dataset_source") or "built_in").strip()
    if dataset_source != "upload" and not require_upload:
        return None, None

    docs_file = request.files.get("dataset_docs")
    questions_file = request.files.get("dataset_questions")
    if not docs_file or not docs_file.filename or not questions_file or not questions_file.filename:
        return None, "Upload both documents JSONL and questions JSONL files."

    docs_error = validate_uploaded_dataset_file(docs_file, "Documents file")
    if docs_error:
        return None, docs_error

    questions_error = validate_uploaded_dataset_file(questions_file, "Questions file")
    if questions_error:
        return None, questions_error

    dataset_root = get_settings().benchmark_root / "user_uploads" / setup_id
    docs_dir = dataset_root / "docs"
    bench_dir = dataset_root / "bench"
    docs_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)
    docs_name = secure_filename(docs_file.filename) or "documents.jsonl"
    docs_file.save(docs_dir / "documents.jsonl")
    questions_file.save(bench_dir / "questions_subset.jsonl")
    (docs_dir / f"original_{docs_name}").write_bytes((docs_dir / "documents.jsonl").read_bytes())
    return str(dataset_root), None


def save_research_setup(payload: dict[str, object]) -> None:
    init_research_setup_store()
    with sqlite3.connect(get_settings().experiment_db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO research_setups(id, payload, created_at) VALUES (?, ?, ?)",
            (payload["id"], json.dumps(payload, default=str), payload["created_at"]),
        )
        conn.commit()


def list_research_setups(limit: int = 8) -> list[dict[str, object]]:
    init_research_setup_store()
    with sqlite3.connect(get_settings().experiment_db_path) as conn:
        rows = conn.execute(
            "SELECT payload FROM research_setups ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    setups = []
    for row in rows:
        try:
            setup = json.loads(row[0])
        except json.JSONDecodeError:
            continue
        setup["created_at_display"] = format_datetime(setup.get("created_at"))
        setups.append(setup)
    return setups


def get_chat_sample_prompts(limit: int = 4) -> list[str]:
    settings = get_settings()
    loader = EnterpriseRagBenchLoader(settings.benchmark_root)
    prompts = [question.question for question in loader.load_questions(max_questions=limit)]
    if prompts:
        return prompts
    return [
        "Which policy or document best answers this enterprise question?",
        "What context would the retriever use to answer a benefits question?",
        "Find documents related to security, access, or internal operations.",
        "Explain the strongest source found for this question.",
    ]


CHAT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
ON chat_messages(session_id, created_at);
"""


def chat_db_path() -> Path:
    return get_settings().experiment_db_path


def init_chat_store() -> None:
    db_path = chat_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(CHAT_SCHEMA_SQL)
        conn.commit()


def current_chat_session_id() -> str:
    init_chat_store()
    chat_session_id = session.get("chat_session_id")
    user_id = session.get("user_id") or get_authorized_data().get("id")
    now = datetime.utcnow().isoformat()

    with sqlite3.connect(chat_db_path()) as conn:
        if chat_session_id:
            row = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = ?",
                (chat_session_id,),
            ).fetchone()
            if row:
                return str(chat_session_id)

        chat_session_id = uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO chat_sessions(id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chat_session_id, user_id, "RAG chat", now, now),
        )
        conn.commit()
    session["chat_session_id"] = chat_session_id
    return chat_session_id


def set_current_chat_session(chat_session_id: str) -> bool:
    init_chat_store()
    with sqlite3.connect(chat_db_path()) as conn:
        row = conn.execute(
            "SELECT id FROM chat_sessions WHERE id = ?",
            (chat_session_id,),
        ).fetchone()
    if not row:
        return False
    session["chat_session_id"] = chat_session_id
    return True


def create_chat_session(title: str = "New RAG chat") -> str:
    init_chat_store()
    chat_session_id = uuid.uuid4().hex
    user_id = session.get("user_id") or get_authorized_data().get("id")
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(chat_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions(id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chat_session_id, user_id, title, now, now),
        )
        conn.commit()
    session["chat_session_id"] = chat_session_id
    return chat_session_id


def clear_current_chat_session() -> None:
    chat_session_id = current_chat_session_id()
    with sqlite3.connect(chat_db_path()) as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (chat_session_id,))
        conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            ("RAG chat", datetime.utcnow().isoformat(), chat_session_id),
        )
        conn.commit()


def list_chat_sessions(limit: int = 12) -> list[dict[str, object]]:
    init_chat_store()
    with sqlite3.connect(chat_db_path()) as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at, COUNT(m.id) AS message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            GROUP BY s.id, s.title, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": row[0],
            "title": row[1] or "RAG chat",
            "created_at": row[2],
            "updated_at": row[3],
            "updated_at_display": format_datetime(row[3]),
            "message_count": row[4],
        }
        for row in rows
    ]


def add_chat_message(
    chat_session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    metadata: dict | None = None,
) -> None:
    now = datetime.utcnow().isoformat()
    message_id = uuid.uuid4().hex
    with sqlite3.connect(chat_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO chat_messages(id, session_id, role, content, sources_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                chat_session_id,
                role,
                content,
                json.dumps(sources or [], default=str),
                json.dumps(metadata or {}, default=str),
                now,
            ),
        )
        if role == "user":
            title = content[:72] or "RAG chat"
            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, chat_session_id),
            )
        else:
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, chat_session_id),
            )
        conn.commit()


def list_chat_messages(chat_session_id: str, limit: int = 40) -> list[dict[str, object]]:
    init_chat_store()
    with sqlite3.connect(chat_db_path()) as conn:
        rows = conn.execute(
            """
            SELECT role, content, sources_json, metadata_json, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (chat_session_id, limit),
        ).fetchall()

    messages = []
    for role, content, sources_json, metadata_json, created_at in reversed(rows):
        try:
            sources = json.loads(sources_json or "[]")
        except json.JSONDecodeError:
            sources = []
        try:
            metadata = json.loads(metadata_json or "{}")
        except json.JSONDecodeError:
            metadata = {}
        messages.append(
            {
                "role": role,
                "content": content,
                "sources": sources,
                "metadata": metadata,
                "created_at": created_at,
            }
        )
    return messages


def latest_chat_sources(messages: list[dict[str, object]]) -> list[dict]:
    for message in reversed(messages):
        sources = message.get("sources")
        if message.get("role") == "assistant" and isinstance(sources, list) and sources:
            return sources
    return []


def format_chat_history(messages: list[dict[str, object]], limit: int = 8) -> str:
    recent = messages[-limit:]
    lines = []
    for message in recent:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:1200]}")
    return "\n\n".join(lines)


def build_chat_context(
    question: str,
    top_k: int,
    strategy: str,
    history: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    payload, error = api_request(
        "POST",
        "/rag/chat",
        timeout=75,
        json={
            "question": question,
            "top_k": top_k,
            "strategy": strategy,
            "history": history or [],
        },
    )
    if error:
        return {
            "error": error,
            "answer": None,
            "documents": [],
            "latency_ms": None,
            "model_label": "Unavailable",
        }

    if not isinstance(payload, dict):
        return {
            "error": "RAG API returned an invalid chat response.",
            "answer": None,
            "documents": [],
            "latency_ms": None,
            "model_label": "Unavailable",
        }

    return {"error": None, **payload}


def normalize_chat_settings(top_k: int | str | None = None, strategy: str | None = None) -> tuple[int, str]:
    if isinstance(top_k, str):
        try:
            top_k = int(top_k)
        except ValueError:
            top_k = None

    stored_top_k = session.get("chat_top_k")
    if top_k is None:
        top_k = stored_top_k if isinstance(stored_top_k, int) else None
    top_k = max(1, min(8, top_k or 4))

    strategy = (strategy or session.get("chat_strategy") or "dense").strip()
    if strategy not in {"dense", "hybrid"}:
        strategy = "dense"
    return top_k, strategy


def store_chat_settings(top_k: int, strategy: str) -> None:
    session["chat_top_k"] = top_k
    session["chat_strategy"] = strategy


def build_ai_chat_context() -> dict[str, object]:
    chat_session_id = current_chat_session_id()
    chat_messages = list_chat_messages(chat_session_id)
    top_k, strategy = normalize_chat_settings()
    return {
        "chat_messages": chat_messages,
        "active_sources": latest_chat_sources(chat_messages),
        "chat_session_id": chat_session_id,
        "chat_sessions": list_chat_sessions(),
        "top_k": top_k,
        "strategy": strategy,
        "sample_prompts": get_chat_sample_prompts(),
    }


def overview_chat_redirect() -> Response:
    return redirect(url_for("index", _anchor="ai-assistant"))


def build_taxonomy_rows(taxonomy: dict | None) -> list[dict[str, object]]:
    labels = {
        "no_relevant_doc_retrieved": "Missing relevant docs",
        "retrieval_noise": "Noisy context",
        "answer_failed_with_context": "Answer failed with context",
        "unknown_or_unlabeled": "Unknown / unlabeled",
    }
    rows = []
    total = sum(int(v or 0) for v in (taxonomy or {}).values()) or 1
    for key, label in labels.items():
        count = int((taxonomy or {}).get(key) or 0)
        rows.append({"key": key, "label": label, "count": count, "pct": (count / total) * 100})
    return rows


def build_dataset_readiness_rows(
    readiness: dict | None,
    *,
    running: bool = False,
) -> list[dict[str, object]]:
    readiness = readiness or {}
    if "ready" in readiness:
        cached_running = bool(running and readiness.get("cached"))
        data_loaded = bool(readiness.get("has_documents") and readiness.get("has_questions"))
        index_loaded = bool(readiness.get("qdrant_ready") or readiness.get("qdrant_busy"))
        if cached_running:
            return [
                {"label": "Data status", "value": "In use"},
                {"label": "Embedding index", "value": "In use" if index_loaded else "Cached"},
                {"label": "Overall status", "value": "Active run"},
            ]
        return [
            {"label": "Data loaded", "value": "Yes" if data_loaded else "No"},
            {"label": "Embedding index", "value": "Loaded" if index_loaded else "Not ready"},
            {"label": "Overall status", "value": "Ready" if readiness.get("ready") else "Needs setup"},
        ]
    return [
        {"label": "Documents", "value": readiness.get("documents", "—")},
        {"label": "Questions", "value": readiness.get("questions", "—")},
        {"label": "Sampled", "value": readiness.get("sampled_questions", "—")},
        {"label": "Holdout", "value": readiness.get("holdout_questions", "—")},
    ]


def build_hypothesis_rows(items: list[dict] | None) -> list[dict[str, object]]:
    rows = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        created_at = parse_iso_datetime(item.get("created_at"))
        rows.append(
            {
                "id": item.get("id") or "—",
                "title": normalize_display_text(item.get("title"), "Untitled"),
                "rationale": normalize_display_text(item.get("rationale"), "—"),
                "expected_impact": normalize_display_text(item.get("expected_impact"), "—"),
                "created_at_display": format_datetime(item.get("created_at")),
                "created_at_sort": created_at,
            }
        )
    rows.sort(key=lambda row: row["created_at_sort"] or datetime.min, reverse=True)
    return rows


def normalize_hypothesis_ids(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        ids = [str(item).strip() for item in value if str(item or "").strip()]
        return list(dict.fromkeys(ids))

    text = str(value or "").strip()
    if not text or text == "—":
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return normalize_hypothesis_ids(parsed)
    for separator in (",", ";"):
        if separator in text:
            return list(dict.fromkeys(part.strip() for part in text.split(separator) if part.strip()))
    return [text]


def find_hypothesis_detail(
    lookup_id: str,
    hypothesis_rows: list[dict[str, object]],
    experiment_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], str | None, str]:
    query = lookup_id.strip()
    if not query:
        return [], [], None, ""

    hypothesis_index = {str(row.get("id") or ""): row for row in hypothesis_rows}
    matched_ids = normalize_hypothesis_ids(query) if query in hypothesis_index else []
    matched_experiment_id = ""

    if not matched_ids:
        for experiment in experiment_rows:
            if str(experiment.get("id") or "") == query:
                matched_ids = normalize_hypothesis_ids(experiment.get("hypothesis_ids") or experiment.get("hypothesis_id"))
                matched_experiment_id = query
                break

    selected_hypotheses = [hypothesis_index[hypothesis_id] for hypothesis_id in matched_ids if hypothesis_id in hypothesis_index]
    if not selected_hypotheses:
        return [], [], f"No hypothesis found for '{query}'.", matched_experiment_id

    selected_ids = {str(row.get("id") or "") for row in selected_hypotheses}
    related_experiments = [
        experiment
        for experiment in experiment_rows
        if selected_ids.intersection(normalize_hypothesis_ids(experiment.get("hypothesis_ids") or experiment.get("hypothesis_id")))
    ]
    return selected_hypotheses, related_experiments, None, matched_experiment_id


def build_failure_chart_rows(results: list[dict] | None) -> list[dict[str, object]]:
    counts = {"Correct": 0, "Incorrect": 0, "Unknown": 0}
    palette = {
        "Correct": "rgba(34, 197, 94, 0.85)",
        "Incorrect": "rgba(239, 68, 68, 0.85)",
        "Unknown": "rgba(156, 163, 175, 0.7)",
    }
    for row in results or []:
        if not isinstance(row, dict):
            continue
        outcome = row.get("is_correct")
        if outcome is True:
            counts["Correct"] += 1
        elif outcome is False:
            counts["Incorrect"] += 1
        else:
            counts["Unknown"] += 1
    return [
        {
            "label": label,
            "count": count,
            "color": palette[label],
        }
        for label, count in counts.items()
    ]


def build_recent_experiment_rows(items: list[dict] | None, limit: int = 2) -> list[dict[str, object]]:
    rows = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        created_at = parse_iso_datetime(item.get("created_at"))
        rows.append(
            {
                "id": item.get("id") or "",
                "name": item.get("name") or "Untitled",
                "run_id": item.get("run_id") or "",
                "run_position": item.get("run_position"),
                "created_at_display": format_datetime(item.get("created_at")),
                "created_at_sort": created_at,
            }
        )
    rows.sort(key=lambda row: row["created_at_sort"] or datetime.min, reverse=True)
    return rows[:limit]


def build_experiment_rows(
    items: list[dict] | None,
    board_by_experiment: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, object]]:
    rows = []
    board_index = board_by_experiment or {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        experiment_id = str(item.get("id") or "")
        config = item.get("retrieval_config") if isinstance(item.get("retrieval_config"), dict) else {}
        question_ids = item.get("question_ids") if isinstance(item.get("question_ids"), list) else []
        score_item = board_index.get(experiment_id, {})
        created_at = parse_iso_datetime(item.get("created_at"))
        hypothesis_ids = normalize_hypothesis_ids(item.get("hypothesis_id"))
        rows.append(
            {
                "id": experiment_id,
                "name": item.get("name") or "Untitled",
                "run_id": item.get("run_id") or "",
                "run_position": item.get("run_position"),
                "hypothesis_ids": hypothesis_ids,
                "hypothesis_id": ", ".join(hypothesis_ids) or "—",
                "strategy": config.get("strategy") if isinstance(config, dict) else None,
                "top_k": config.get("top_k") if isinstance(config, dict) else None,
                "question_count": len(question_ids),
                "created_at_display": format_datetime(item.get("created_at")),
                "created_at_sort": created_at,
                "config_json": serialize_for_template(config),
                "score": to_float(score_item.get("score")),
                "delta_vs_baseline": to_float(score_item.get("delta_vs_baseline")),
                "accepted": score_item.get("accepted"),
            }
        )
    rows.sort(key=lambda row: row["created_at_sort"] or datetime.min, reverse=True)
    return rows


def build_experiment_run_groups(
    experiment_rows: list[dict[str, object]],
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Group experiments by persisted run_id, with a timestamp fallback for legacy data."""
    sorted_rows = sorted(
        experiment_rows,
        key=lambda row: row["created_at_sort"] or datetime.min,
    )
    grouped: dict[str, dict[str, object]] = {}
    legacy_run_count = 0
    legacy_key = ""
    legacy_latest: datetime | None = None
    legacy_gap = timedelta(minutes=45)

    for row in sorted_rows:
        run_id = str(row.get("run_id") or "")
        created_at = row.get("created_at_sort")
        created_at_dt = created_at if isinstance(created_at, datetime) else None

        if run_id:
            key = run_id
            inferred = False
        else:
            if not legacy_key or (
                created_at_dt is not None
                and legacy_latest is not None
                and created_at_dt - legacy_latest > legacy_gap
            ):
                legacy_run_count += 1
                legacy_key = f"legacy_run_{legacy_run_count}"
            key = legacy_key or "legacy_run_1"
            inferred = True

        legacy_latest = created_at_dt if created_at_dt is not None else legacy_latest
        group = grouped.setdefault(
            key,
            {
                "id": key,
                "label": key.replace("_", " ").title() if inferred else key,
                "inferred": inferred,
                "experiments": [],
                "experiment_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "latest_created_at": None,
                "latest_created_at_display": "—",
                "latest_experiment": None,
            },
        )
        experiments = group["experiments"]
        if isinstance(experiments, list):
            experiments.append(row)
        group["experiment_count"] = int(group["experiment_count"] or 0) + 1
        if row.get("accepted") is True:
            group["accepted_count"] = int(group["accepted_count"] or 0) + 1
        elif row.get("accepted") is False:
            group["rejected_count"] = int(group["rejected_count"] or 0) + 1

        latest_created_at = group["latest_created_at"]
        if (
            created_at_dt is not None
            and (not isinstance(latest_created_at, datetime) or created_at_dt >= latest_created_at)
        ):
            group["latest_created_at"] = created_at_dt
            group["latest_created_at_display"] = row.get("created_at_display") or "—"
            group["latest_experiment"] = row

    groups = list(grouped.values())
    for group in groups:
        experiments = group.get("experiments")
        if isinstance(experiments, list):
            experiments.sort(key=lambda row: row["created_at_sort"] or datetime.min, reverse=True)
            positions = [
                int(row["run_position"])
                for row in experiments
                if isinstance(row.get("run_position"), int)
            ]
            if positions:
                group["label"] = f"Run {str(group['id']).replace('run_', '')}"
            elif group.get("inferred"):
                group["label"] = f"Inferred {group['label']}"

    groups.sort(
        key=lambda group: group["latest_created_at"]
        if isinstance(group.get("latest_created_at"), datetime)
        else datetime.min,
        reverse=True,
    )
    return groups[:limit] if limit else groups


def build_metric_rows(metrics: dict | None) -> list[dict[str, str]]:
    payload = metrics if isinstance(metrics, dict) else {}
    definitions = [
        ("Total questions", "total_questions", 0),
        ("Answered questions", "answered_questions", 0),
        ("Recall@k", "recall_at_k", 3),
        ("Precision@k", "precision_at_k", 3),
        ("Answer correctness", "answer_correctness", 3),
        ("Avg latency (ms)", "avg_latency_ms", 2),
        ("Invalid extra docs rate", "invalid_extra_docs_rate", 3),
    ]
    rows = []
    for label, key, decimals in definitions:
        value = payload.get(key)
        if decimals == 0:
            display = str(int(value)) if isinstance(value, (int, float)) else "—"
        else:
            numeric = to_float(value)
            display = f"{numeric:.{decimals}f}" if numeric is not None else "—"
        rows.append({"label": label, "display": display})
    return rows


def build_question_result_rows(
    question_results: list[dict] | None,
) -> tuple[list[dict[str, object]], float | None, int]:
    rows = []
    correct_count = 0
    evaluated_count = 0

    for row in question_results or []:
        if not isinstance(row, dict):
            continue
        is_correct = row.get("is_correct")
        if is_correct is True:
            outcome = "Correct"
            correct_count += 1
            evaluated_count += 1
        elif is_correct is False:
            outcome = "Incorrect"
            evaluated_count += 1
        else:
            outcome = "Unknown"

        answer = str(row.get("answer") or "").strip()
        if len(answer) > 140:
            answer = f"{answer[:137]}..."
        document_ids = row.get("document_ids") if isinstance(row.get("document_ids"), list) else []

        rows.append(
            {
                "question_id": row.get("question_id") or "—",
                "outcome": outcome,
                "latency_ms": to_float(row.get("latency_ms")),
                "document_count": len(document_ids),
                "answer_preview": answer or "—",
            }
        )

    accuracy_pct = (correct_count / evaluated_count * 100) if evaluated_count else None
    return rows, accuracy_pct, evaluated_count


def build_experiment_detail_context(experiment: dict | None) -> dict[str, object]:
    payload = experiment if isinstance(experiment, dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    question_rows, question_accuracy_pct, evaluated_count = build_question_result_rows(payload.get("question_results"))
    return {
        "composite_score": to_float(payload.get("composite_score")),
        "baseline_score": to_float(payload.get("baseline_score")),
        "delta_vs_baseline": to_float(payload.get("delta_vs_baseline")),
        "status_label": payload.get("status") or "—",
        "accepted": payload.get("accepted"),
        "created_at_display": format_datetime(payload.get("created_at")),
        "metric_rows": build_metric_rows(metrics),
        "question_rows": question_rows,
        "question_accuracy_pct": question_accuracy_pct,
        "question_evaluated_count": evaluated_count,
    }


def safe_oauth_call(method_name: str, *args, **kwargs):
    method = getattr(kinde_oauth, method_name, None)
    if not callable(method):
        return None
    try:
        return method(*args, **kwargs)
    except Exception as exc:
        logger.debug("Kinde OAuth call failed for %s: %s", method_name, exc)
        return None


def fetch_management_users(limit: int = 25) -> tuple[list[dict[str, object]], str | None]:
    host = (os.getenv("KINDE_ISSUER_URL") or os.getenv("KINDE_HOST") or "").rstrip("/")
    client_id = os.getenv("MGMT_API_CLIENT_ID") or os.getenv("KINDE_MANAGEMENT_CLIENT_ID")
    client_secret = os.getenv("MGMT_API_CLIENT_SECRET") or os.getenv("KINDE_MANAGEMENT_CLIENT_SECRET")
    audience = os.getenv("MGMT_API_AUDIENCE") or (f"{host}/api" if host else "")
    users_url = os.getenv("MGMT_API_USERS_URL") or (f"{host}/api/v1/users" if host else "")

    if not host or not client_id or not client_secret:
        return [], "Management API is not configured."

    try:
        token_response = requests.post(
            f"{host}/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "audience": audience,
            },
            timeout=10,
        )
        token_response.raise_for_status()
        token_payload = token_response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Failed to request management API token: %s", exc)
        return [], "Failed to request management API token."

    access_token = token_payload.get("access_token")
    if not access_token:
        return [], "Management API token response was missing access_token."

    try:
        users_response = requests.get(
            users_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        users_response.raise_for_status()
        payload = users_response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Failed to query management API users: %s", exc)
        return [], "Failed to query management API users."

    if isinstance(payload, dict):
        users_payload = payload.get("users")
    elif isinstance(payload, list):
        users_payload = payload
    else:
        users_payload = []

    users = []
    for item in users_payload or []:
        if not isinstance(item, dict):
            continue
        users.append(
            {
                "first_name": item.get("first_name") or item.get("given_name") or "—",
                "last_name": item.get("last_name") or item.get("family_name") or "—",
                "total_sign_ins": item.get("total_sign_ins", "—"),
            }
        )
        if len(users) >= limit:
            break

    return users, None


def build_config_items(config: dict | None, baseline: dict | None = None) -> list[dict[str, object]]:
    skip = {"evaluation_mode", "extra"}
    items = []
    for key, value in (config or {}).items():
        if key in skip:
            continue
        items.append(
            {
                "key": key,
                "value": value,
                "changed": baseline is not None and baseline.get(key) != value,
            }
        )
    return items


def parse_per_type_summary(summary: str | None) -> list[dict[str, object]]:
    rows = []
    for part in (summary or "").split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            name, rest = part.split(":", 1)
            tokens = rest.strip().split()
            recall_value = float(tokens[0].split("=")[1])
            precision_value = float(tokens[1].split("=")[1])
            questions = tokens[2].strip("()")
            rows.append(
                {
                    "name": name.strip(),
                    "recall": recall_value,
                    "precision": precision_value,
                    "questions": questions,
                }
            )
        except (IndexError, ValueError):
            continue
    return rows


def build_score_history_rows(score_history: list[dict] | None, initial_baseline: float | None) -> list[dict[str, object]]:
    history = score_history or []
    numeric_values = []
    if initial_baseline is not None:
        numeric_values.append(initial_baseline)

    for entry in history:
        score = to_float(entry.get("score"))
        baseline = to_float(entry.get("baseline"))
        if score is not None:
            numeric_values.append(score)
        if baseline is not None:
            numeric_values.append(baseline)

    max_value = max(numeric_values, default=0.0)
    if max_value <= 0:
        max_value = 1.0

    baseline_pct = max(0.0, (initial_baseline / max_value) * 100) if initial_baseline is not None else None

    rows = []
    for entry in history:
        iteration = entry.get("iteration")
        score = to_float(entry.get("score"))
        baseline = to_float(entry.get("baseline"))
        accepted = bool(entry.get("accepted"))
        reason = str(entry.get("reason") or "").strip()
        validation_delta = to_float(entry.get("validation_delta"))
        rows.append(
            {
                "iteration": iteration,
                "label": f"Iter {iteration}",
                "score": score,
                "baseline": baseline,
                "validation_delta": validation_delta,
                "accepted": accepted,
                "result_label": "Accepted" if accepted else "Rejected",
                "reason": reason,
                "score_pct": max(0.0, ((score or 0.0) / max_value) * 100),
                "baseline_pct": max(0.0, ((baseline or 0.0) / max_value) * 100),
                "initial_baseline_pct": baseline_pct,
            }
        )
    return rows


def build_journey_entries(score_history: list[dict] | None, baseline_config: dict | None) -> list[dict[str, object]]:
    entries = []
    for entry in score_history or []:
        score = to_float(entry.get("score"))
        baseline = to_float(entry.get("baseline"))
        delta = score - baseline if score is not None and baseline is not None else None
        accepted = bool(entry.get("accepted"))
        entries.append(
            {
                "iteration": entry.get("iteration"),
                "experiment_id": str(entry.get("experiment_id") or ""),
                "hypothesis_id": (normalize_hypothesis_ids(entry.get("hypothesis_id")) or [""])[0],
                "score": score,
                "baseline": baseline,
                "delta": delta,
                "accepted": accepted,
                "icon": "✓" if accepted else "×",
                "hypothesis": normalize_display_text(entry.get("hypothesis")),
                "config_items": build_config_items(entry.get("config"), baseline_config),
            }
        )
    return entries


def build_code_history_entries(code_history: list[dict] | None, baseline_config: dict | None) -> list[dict[str, object]]:
    entries = []
    for entry in code_history or []:
        if not isinstance(entry, dict):
            continue
        proposed_config = entry.get("proposed_config") if isinstance(entry.get("proposed_config"), dict) else None
        entries.append(
            {
                **entry,
                "experiment_id": str(entry.get("experiment_id") or ""),
                "hypothesis_id": (normalize_hypothesis_ids(entry.get("hypothesis_id")) or [""])[0],
                "hypothesis": normalize_display_text(entry.get("hypothesis")),
                "diff_summary": normalize_display_text(entry.get("diff_summary")),
                "config_items": build_config_items(proposed_config, baseline_config),
            }
        )
    return entries


def build_leaderboard_rows(board: list[dict] | None) -> list[dict[str, object]]:
    rows = []
    max_score = max((to_float(item.get("score")) or 0.0 for item in board or []), default=0.0)
    if max_score <= 0:
        max_score = 1.0

    for index, item in enumerate(board or [], start=1):
        score = to_float(item.get("score"))
        delta = to_float(item.get("delta_vs_baseline"))
        rows.append(
            {
                **item,
                "rank": index,
                "score_value": score,
                "delta_value": delta,
                "score_bar_pct": max(0.0, ((score or 0.0) / max_score) * 100),
            }
        )
    return rows


def has_google_api_key_for_session() -> bool:
    if bool(get_session_runtime_google_api_key()):
        return True
    return bool(get_settings().google_api_key.strip())


def api_request(method: str, path: str, **kwargs):
    url = f"{get_dashboard_api_base()}/{path.lstrip('/')}"
    timeout = kwargs.pop("timeout", 10)
    headers = dict(kwargs.pop("headers", {}) or {})
    runtime_api_key = ""
    if has_request_context():
        runtime_api_key = get_session_runtime_google_api_key()
    if runtime_api_key:
        headers["X-Google-Api-Key"] = runtime_api_key
    if headers:
        kwargs["headers"] = headers
    started_at = time.perf_counter()

    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.warning("Dashboard API request failed for %s after %.1fms: %s", url, elapsed_ms, exc)
        return None, f"Could not reach API at {url}."
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    if elapsed_ms > 500:
        logger.warning(
            "Dashboard API request slow method=%s path=%s status=%s elapsed_ms=%.1f",
            method,
            path,
            response.status_code,
            elapsed_ms,
        )

    if not response.ok:
        detail = None
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail")

        if not detail:
            detail = response.text.strip() or response.reason

        return None, detail or f"API request failed with status {response.status_code}."

    try:
        return response.json(), None
    except ValueError:
        return None, "API returned invalid JSON."


def current_research_status_params() -> dict[str, str] | None:
    run_id = str(session.get("research_run_id") or "").strip()
    return {"run_id": run_id} if run_id else None


def get_dashboard_status():
    live_mode, refresh_every = get_dashboard_preferences()
    status_params = current_research_status_params()
    if status_params:
        status, error = api_request("GET", "/research/status", params=status_params)
    else:
        status, error = None, None
    if isinstance(status, dict) and status.get("run_id"):
        session["research_run_id"] = status["run_id"]
        for key in ("latest_summary", "planner_rationale", "final_report", "recommendation", "validation_summary"):
            status[key] = normalize_display_text(status.get(key))
    is_running = bool(status and status.get("running"))
    show_running_animation = bool(session.get("research_pending") or is_running)

    initial_baseline = to_float(status.get("initial_baseline_score")) if status else None
    best_score = to_float(status.get("best_score")) if status else None
    if initial_baseline is not None and initial_baseline < 0:
        initial_baseline = None
    if best_score is not None and best_score < 0:
        best_score = None
    best_delta = None
    improvement_pct = None
    if initial_baseline is not None and best_score is not None:
        best_delta = best_score - initial_baseline
        if initial_baseline > 0:
            improvement_pct = (best_delta / initial_baseline) * 100

    baseline_config = status.get("baseline_config") if status else None
    candidate_config = status.get("candidate_config") if status else None
    score_history = status.get("score_history") if status else []

    if is_running:
        session["research_pending"] = True
    elif session.get("research_pending"):
        session.pop("research_pending", None)

    status_research_mode = status.get("research_mode") if status else None
    if is_running:
        research_mode = status_research_mode or session.get("research_mode") or "config"
    else:
        research_mode = session.get("research_mode") or status_research_mode or "config"

    status_max_iterations = status.get("max_iterations") if status else None
    if is_running and status_max_iterations:
        selected_max_iterations = int(status_max_iterations)
    else:
        selected_max_iterations = int(session.get("max_iterations") or status_max_iterations or 5)
    selected_max_iterations = max(1, min(MAX_RESEARCH_ITERATIONS, selected_max_iterations))
    code_history = build_code_history_entries(status.get("code_history", []) if status else [], baseline_config)
    karpathy_branch = status.get("karpathy_branch", "") if status else ""
    initial_pipeline_code = status.get("initial_pipeline_code", "") if status else ""
    current_pipeline_code = status.get("current_pipeline_code", "") if status else ""
    proposed_code = status.get("proposed_code", "") if status else ""
    pipeline_display_code = current_pipeline_code
    pipeline_display_label = "Current accepted code"

    pipeline_diff = ""
    if proposed_code and proposed_code != current_pipeline_code:
        pipeline_display_code = proposed_code
        pipeline_display_label = "Proposed candidate"
        pipeline_diff = "".join(difflib.unified_diff(
            current_pipeline_code.splitlines(keepends=True),
            proposed_code.splitlines(keepends=True),
            fromfile="pipeline.py (current)",
            tofile="pipeline.py (proposed)",
            n=3,
        ))
    elif initial_pipeline_code and current_pipeline_code and initial_pipeline_code != current_pipeline_code:
        pipeline_diff = "".join(difflib.unified_diff(
            initial_pipeline_code.splitlines(keepends=True),
            current_pipeline_code.splitlines(keepends=True),
            fromfile="pipeline.py (original)",
            tofile="pipeline.py (current)",
            n=3,
        ))

    return {
        "live_mode": live_mode,
        "refresh_every": refresh_every,
        "status": status,
        "error": error,
        "show_running_animation": show_running_animation,
        "initial_baseline": initial_baseline,
        "best_score": best_score,
        "best_delta": best_delta,
        "improvement_pct": improvement_pct,
        "baseline_config_items": build_config_items(baseline_config),
        "candidate_config_items": build_config_items(candidate_config, baseline_config),
        "score_history_rows": build_score_history_rows(score_history, initial_baseline),
        "journey_entries": build_journey_entries(score_history, baseline_config),
        "per_type_rows": parse_per_type_summary(status.get("per_type_summary") if status else None),
        "taxonomy_rows": build_taxonomy_rows(status.get("failure_taxonomy") if status else None),
        "dataset_readiness_rows": build_dataset_readiness_rows(
            status.get("dataset_readiness") if status else None,
            running=is_running,
        ),
        "recommendation": status.get("recommendation") if status else None,
        "validation_summary": status.get("validation_summary") if status else None,
        "final_report_html": render_markdown(status.get("final_report") if status else None),
        "tried_config_count": status.get("tried_config_count") if status else 0,
        "rejected_config_count": status.get("rejected_config_count") if status else 0,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question_type_options": get_question_type_options(),
        "research_setups": list_research_setups(),
        "default_starting_config": build_starting_config_for_focus("all"),
        "has_uploaded_dataset": bool(session.get("uploaded_benchmark_root")),
        "use_uploaded_dataset": bool(session.get("use_uploaded_dataset") and session.get("uploaded_benchmark_root")),
        "uploaded_dataset_label": session.get("uploaded_dataset_label") or "Uploaded dataset",
        "selected_max_iterations": selected_max_iterations,
        # Karpathy mode
        "research_mode": research_mode,
        "code_history": code_history,
        "karpathy_branch": karpathy_branch,
        "initial_pipeline_code": initial_pipeline_code,
        "current_pipeline_code": current_pipeline_code,
        "proposed_code": proposed_code,
        "pipeline_display_code": pipeline_display_code,
        "pipeline_display_label": pipeline_display_label,
        "pipeline_diff": pipeline_diff,
    }


def get_sidebar_status_snapshot() -> dict[str, object]:
    if not kinde_oauth.is_authenticated():
        return {}

    status_params = current_research_status_params()
    if not status_params:
        return {
            "reachable": True,
            "state": "Ready",
            "state_class": "light",
            "phase": "idle",
            "progress_pct": 0,
            "message": "No active research run in this session.",
            "last_error": None,
            "accepted": 0,
            "rejected": 0,
            "best_score": None,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        }

    status, error = api_request(
        "GET",
        "/research/status",
        timeout=2,
        params={**status_params, "detail": "summary"},
    )
    if error or not isinstance(status, dict):
        if session.get("research_pending"):
            return {
                "reachable": True,
                "state": "Updating",
                "state_class": "warning",
                "phase": "syncing",
                "progress_pct": 0,
                "message": "Refreshing run status...",
                "last_error": None,
                "accepted": 0,
                "rejected": 0,
                "best_score": None,
                "updated_at": datetime.now().strftime("%H:%M:%S"),
            }
        return {
            "reachable": False,
            "state": "Offline",
            "state_class": "danger",
            "phase": "API unavailable",
            "progress_pct": 0,
            "message": error or "No status available.",
            "last_error": None,
            "accepted": 0,
            "rejected": 0,
            "best_score": None,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        }

    running = bool(status.get("running"))
    iteration = status.get("iteration") or 0
    max_iterations = status.get("max_iterations") or 0
    progress_pct = int((iteration / max(max_iterations, 1)) * 100) if max_iterations else 0
    last_error = status.get("last_error")
    final_report = status.get("final_report")
    phase = normalize_display_text(status.get("phase"))
    stopped = phase == "stopped"
    state = "Running" if running else "Stopped" if stopped else "Complete" if final_report else "Ready"
    state_class = "success" if running else "warning" if stopped else "link" if final_report else "light"

    return {
        "reachable": True,
        "state": state,
        "state_class": state_class,
        "phase": phase if running or final_report or stopped else "",
        "iteration": iteration,
        "max_iterations": max_iterations,
        "progress_pct": max(0, min(100, progress_pct)),
        "message": normalize_display_text(status.get("latest_summary"), "No updates yet."),
        "last_error": last_error,
        "accepted": status.get("accepted_experiments") or 0,
        "rejected": status.get("rejected_experiments") or 0,
        "best_score": to_float(status.get("best_score")),
        "updated_at": datetime.now().strftime("%H:%M:%S"),
    }


@app.context_processor
def inject_template_context():
    user_data = get_authorized_data()
    return {
        "current_year": date.today().year,
        "current_endpoint": request.endpoint,
        "nav_items": build_nav_items(),
        "user_initials": get_user_initials(user_data),
        "user_display_name": get_user_display_name(user_data),
        "dashboard_live_mode": normalize_bool(session.get("live_mode"), default=False),
        "dashboard_refresh_every": normalize_refresh_seconds(session.get("refresh_every")),
        "sidebar_status": get_sidebar_status_snapshot(),
        "has_google_api_key": has_google_api_key_for_session(),
        **user_data,
    }


@app.before_request
def protect_dashboard_routes():
    ensure_active_event_loop()

    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
        return None

    if kinde_oauth.is_authenticated():
        return None

    if request.endpoint in BACKGROUND_ENDPOINTS or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return unauthenticated_background_response()

    remember_post_login_redirect()
    return redirect(url_for("login"))


@app.route("/")
def index():
    if kinde_oauth.is_authenticated():
        dashboard = get_dashboard_status()
        dashboard["selected_question_focus"] = session.get("selected_question_focus") or "all"
        dashboard.update(build_ai_chat_context())
        return render_template("home.html", **dashboard)
    return render_template("logged_out.html")


@app.route("/up")
def up():
    return "OK", 200


@app.route("/dashboard/preferences", methods=["POST"])
def dashboard_preferences():
    session["live_mode"] = request.form.get("live_mode") == "1"
    session["refresh_every"] = normalize_refresh_seconds(request.form.get("refresh_every"))
    flash("Dashboard settings updated.", "success")
    next_url = (request.form.get("next") or "").strip() or url_for("index")
    return redirect(next_url)


@app.route("/dashboard/status")
def dashboard_status_snapshot():
    live_mode, refresh_every = get_dashboard_preferences()
    sidebar_status = get_sidebar_status_snapshot()
    return jsonify(
        {
            "live_mode": live_mode,
            "refresh_every": refresh_every,
            "should_poll": live_mode or sidebar_status.get("state") in {"Running", "Updating"},
            "sidebar_status": sidebar_status,
        }
    )


@app.route("/dashboard/api-key", methods=["POST"])
def dashboard_api_key():
    api_key = (request.form.get("api_key") or "").strip()
    next_url = (request.form.get("next") or "").strip() or url_for("index")
    if not api_key:
        clear_runtime_google_api_key()
        flash("API key cleared for this dashboard session. The default server key will be used.", "success")
        return redirect(next_url)

    user_id = get_authenticated_user_id()
    if not user_id:
        flash("Could not apply API key because the current user could not be verified.", "error")
        return redirect(next_url)

    session["runtime_google_api_key"] = api_key
    session["runtime_google_api_key_user_id"] = user_id
    flash("API key applied for this dashboard session.", "success")
    return redirect(next_url)


@app.route("/save_focus", methods=["POST"])
def save_focus():
    question_focus = (request.form.get("question_focus") or "all").strip()
    session["selected_question_focus"] = question_focus

    action = (request.form.get("action") or "save_focus").strip()
    if action == "upload_dataset":
        setup_id = uuid.uuid4().hex
        benchmark_root, upload_error = save_uploaded_research_dataset(setup_id, require_upload=True)
        if upload_error:
            flash(upload_error, "error")
            return redirect(url_for("index"))
        session["uploaded_benchmark_root"] = benchmark_root
        session["uploaded_dataset_label"] = f"Uploaded dataset · {datetime.now().strftime('%b %d, %H:%M')}"
        session["use_uploaded_dataset"] = True
        flash("Dataset uploaded and selected for the next experiment.", "success")
        return redirect(url_for("index"))

    has_uploaded_dataset = bool(session.get("uploaded_benchmark_root"))
    session["use_uploaded_dataset"] = has_uploaded_dataset and request.form.get("use_uploaded_dataset") == "1"
    dataset_note = " using your uploaded dataset" if session.get("use_uploaded_dataset") else ""
    flash(f"Question focus saved: {question_focus.replace('_', ' ').title()}{dataset_note}.", "success")
    return redirect(url_for("index"))


@app.route("/save_research_mode", methods=["POST"])
def save_research_mode():
    mode = (request.form.get("research_mode") or "config").strip()
    if mode not in ("config", "karpathy"):
        mode = "config"
    session["research_mode"] = mode

    max_iterations = request.form.get("max_iterations", type=int)
    if max_iterations is not None:
        session["max_iterations"] = max(1, min(MAX_RESEARCH_ITERATIONS, max_iterations))

    return "", 204


@app.route("/start_research", methods=["POST"])
def start_research():
    max_iterations = request.form.get("max_iterations", type=int) or 5
    max_iterations = max(1, min(MAX_RESEARCH_ITERATIONS, max_iterations))
    refresh_every = normalize_refresh_seconds(request.form.get("refresh", type=int))
    live_mode = request.form.get("live") == "1"
    question_focus = (request.form.get("question_focus") or "all").strip()
    use_uploaded_dataset = bool(session.get("use_uploaded_dataset") and session.get("uploaded_benchmark_root"))
    dataset_source = "upload" if use_uploaded_dataset else (request.form.get("dataset_source") or "built_in").strip()
    setup_id = uuid.uuid4().hex
    starting_config = build_starting_config_for_focus(question_focus)
    benchmark_root = session.get("uploaded_benchmark_root") if use_uploaded_dataset else None
    upload_error = None

    if not use_uploaded_dataset:
        benchmark_root, upload_error = save_uploaded_research_dataset(setup_id)

    if upload_error:
        flash(upload_error, "error")
        return redirect(url_for("index"))

    question_label = next(
        (item["label"] for item in get_question_type_options() if item["value"] == question_focus),
        question_focus.replace("_", " ").title() if question_focus != "all" else "Generic mix",
    )
    dataset_label = session.get("uploaded_dataset_label") if dataset_source == "upload" else "Built-in test dataset"
    setup_payload = {
        "id": setup_id,
        "question_focus": question_focus,
        "question_focus_label": question_label,
        "dataset_source": dataset_source,
        "dataset_label": dataset_label,
        "benchmark_root": benchmark_root,
        "starting_config": starting_config,
        "created_at": datetime.utcnow().isoformat(),
    }
    save_research_setup(setup_payload)

    session["live_mode"] = live_mode
    session["refresh_every"] = refresh_every
    session["selected_question_focus"] = question_focus
    session["max_iterations"] = max_iterations

    research_mode = (request.form.get("research_mode") or "config").strip()
    if research_mode not in ("config", "karpathy"):
        research_mode = "config"
    session["research_mode"] = research_mode

    payload, error = api_request(
        "POST",
        "/research/start",
        params={
            "max_iterations": max_iterations,
            "question_focus": question_focus,
            "benchmark_root": benchmark_root or "",
            "research_setup_id": setup_id,
            "starting_config_json": json.dumps(starting_config),
            "research_mode": research_mode,
        },
    )
    if error:
        flash(error, "error")
        session.pop("research_pending", None)
    elif payload and payload.get("status") == "already_running":
        session["research_pending"] = True
        flash("Research loop is already running.", "warning")
    else:
        if payload and payload.get("run_id"):
            session["research_run_id"] = payload["run_id"]
        session["research_pending"] = True
        flash(f"Started tuned loop for {question_label} with {dataset_label}.", "success")

    return redirect(url_for("experiments"))


@app.route("/stop_research", methods=["POST"])
def stop_research():
    run_id = str(session.get("research_run_id") or "").strip()
    params = {"run_id": run_id} if run_id else {}
    payload, error = api_request("POST", "/research/stop", params=params)
    session.pop("research_pending", None)
    if error:
        flash(f"Could not stop experiment: {error}", "error")
    elif payload and payload.get("status") == "stopped":
        flash("Experiment run stopped.", "warning")
    else:
        flash("Stop request sent.", "warning")
    return redirect(url_for("experiments"))


@app.route("/commit-pipeline", methods=["POST"])
def commit_pipeline():
    message = (request.form.get("message") or "").strip() or "Accepted pipeline from dashboard"
    payload, error = api_request(
        "POST",
        "/research/commit-pipeline",
        params={"message": message},
    )
    if error:
        flash(f"Commit failed: {error}", "error")
    elif payload and payload.get("status") == "committed":
        flash(f"Pipeline committed on {payload.get('branch', 'branch')}.", "success")
    else:
        flash("Unexpected response from commit.", "warning")
    return redirect(url_for("experiments"))


@app.route("/signout")
def dashboard_logout():
    user_id = session.get("user_id")
    session.clear()
    logout_redirect = os.getenv("LOGOUT_REDIRECT_URL") or url_for("index", _external=True)
    logout_url = ensure_active_event_loop().run_until_complete(
        kinde_oauth.logout(
            user_id,
            {"post_logout_redirect_uri": logout_redirect},
        )
    )
    return redirect(logout_url)


@app.route("/experiments")
def experiments():
    dashboard = get_dashboard_status()

    question_focus = session.get("selected_question_focus") or "all"
    focus_label = next(
        (opt["label"] for opt in dashboard["question_type_options"] if opt["value"] == question_focus),
        "Generic mix",
    )

    return render_template(
        "experiments.html",
        **dashboard,
        selected_question_focus=question_focus,
        selected_question_focus_label=focus_label,
    )


@app.route("/hypotheses", methods=["GET", "POST"])
def hypotheses():
    items, error = api_request("GET", "/hypotheses")
    experiment_items, experiment_error = api_request("GET", "/experiments")
    rows = build_hypothesis_rows(items or [])
    experiment_rows = build_experiment_rows(experiment_items or [])

    if request.method == "POST":
        lookup_id = (request.form.get("lookup_id") or "").strip()
    else:
        lookup_id = (
            request.args.get("hypothesis_id")
            or request.args.get("experiment_id")
            or request.args.get("lookup_id")
            or ""
        ).strip()

    selected_hypotheses, related_experiments, search_error, matched_experiment_id = find_hypothesis_detail(
        lookup_id,
        rows,
        experiment_rows,
    )
    return render_template(
        "hypotheses.html",
        hypotheses=rows,
        experiments=experiment_rows,
        selected_hypotheses=selected_hypotheses,
        related_experiments=related_experiments,
        lookup_id=lookup_id,
        matched_experiment_id=matched_experiment_id,
        error=error,
        experiment_error=experiment_error,
        search_error=search_error,
    )


@app.route("/experiments/new")
def new_experiment():
    return redirect(url_for("experiments"))


@app.route("/leaderboards")
def leaderboards():
    board, error = api_request("GET", "/leaderboard")
    board_rows = build_leaderboard_rows(board or [])
    worst = None
    results = []
    detail_error = None

    if board_rows:
        scored_board = [item for item in board_rows if isinstance(item.get("score_value"), (int, float))]
        if scored_board:
            worst = min(scored_board, key=lambda item: item["score_value"])
            detail, detail_error = api_request(
                "GET",
                f"/experiments/{quote(str(worst['experiment_id']), safe='')}",
            )
            if detail:
                results = detail.get("question_results", [])
                if not worst.get("failure_analysis"):
                    worst["failure_analysis"] = detail.get("failure_analysis")

    return render_template(
        "leaderboards.html",
        board=board or [],
        board_rows=board_rows,
        error=error,
        worst=worst,
        results=results,
        detail_error=detail_error,
        failure_chart_rows=build_failure_chart_rows(results),
    )


@app.route("/export/<kind>")
def export_dashboard_artifact(kind: str):
    status_params = current_research_status_params()
    if not status_params:
        flash("No research run is active in this session.", "error")
        return redirect(url_for("index"))
    status, error = api_request("GET", "/research/status", params=status_params)
    if error or not isinstance(status, dict):
        flash(error or "No research status available to export.", "error")
        return redirect(url_for("index"))

    if kind == "report":
        body = status.get("final_report") or status.get("latest_summary") or "No report available yet."
        filename = "research_report.txt"
        mimetype = "text/plain"
    elif kind == "config":
        body = json.dumps(status.get("candidate_config") or status.get("baseline_config") or {}, indent=2, default=str)
        filename = "current_config.json"
        mimetype = "application/json"
    elif kind == "failures":
        body = json.dumps(status.get("failure_examples") or [], indent=2, default=str)
        filename = "failure_examples.json"
        mimetype = "application/json"
    else:
        flash("Unknown export type.", "error")
        return redirect(url_for("index"))

    return Response(
        body,
        mimetype=mimetype,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/ai-chat", methods=["GET", "POST"])
def ai_chat():
    requested_session_id = (request.args.get("session_id") or "").strip()
    if requested_session_id and not set_current_chat_session(requested_session_id):
        flash("Chat session not found.", "warning")
        return overview_chat_redirect()

    if request.method == "GET":
        return overview_chat_redirect()

    chat_session_id = current_chat_session_id()
    top_k, strategy = normalize_chat_settings(
        request.form.get("top_k", type=int),
        request.form.get("strategy"),
    )
    store_chat_settings(top_k, strategy)

    if request.form.get("action") == "clear":
        clear_current_chat_session()
        flash("AI chat history cleared.", "success")
        return overview_chat_redirect()

    question = (request.form.get("question") or "").strip()
    if not question:
        flash("Ask a question to test retrieval.", "warning")
        return overview_chat_redirect()

    try:
        prior_messages = list_chat_messages(chat_session_id)
        add_chat_message(chat_session_id, "user", question)
        chat_result = build_chat_context(
            question,
            top_k=top_k,
            strategy=strategy,
            history=prior_messages,
        )
        error = chat_result.get("error")
        if error:
            add_chat_message(
                chat_session_id,
                "assistant",
                str(error),
                sources=chat_result.get("documents") if isinstance(chat_result.get("documents"), list) else [],
                metadata={"error": True, "strategy": strategy, "top_k": top_k},
            )
        elif chat_result.get("answer"):
            add_chat_message(
                chat_session_id,
                "assistant",
                str(chat_result["answer"]),
                sources=chat_result.get("documents") if isinstance(chat_result.get("documents"), list) else [],
                metadata={
                    "latency_ms": chat_result.get("latency_ms"),
                    "model_label": chat_result.get("model_label"),
                    "strategy": strategy,
                    "top_k": top_k,
                },
            )
    except Exception as exc:
        logger.exception("RAG chat test failed")
        error = f"RAG chat test failed: {exc}"
        add_chat_message(
            chat_session_id,
            "assistant",
            error,
            metadata={"error": True, "strategy": strategy, "top_k": top_k},
        )

    return overview_chat_redirect()


@app.route("/ai-chat/new", methods=["POST"])
def new_ai_chat():
    create_chat_session()
    flash("Started a new AI chat.", "success")
    return redirect(url_for("index", _anchor="ai-assistant"))


@app.route("/ai-chat/export")
def export_ai_chat():
    chat_session_id = (request.args.get("session_id") or session.get("chat_session_id") or "").strip()
    if not chat_session_id or not set_current_chat_session(chat_session_id):
        flash("Chat session not found.", "error")
        return overview_chat_redirect()

    messages = list_chat_messages(chat_session_id, limit=500)
    body = json.dumps(
        {
            "session_id": chat_session_id,
            "messages": messages,
            "exported_at": datetime.utcnow().isoformat(),
        },
        indent=2,
        default=str,
    )
    return Response(
        body,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=ai_chat_history.json"},
    )


@app.route("/failure_analysis")
def failure_analysis():
    return redirect(url_for("leaderboards", _anchor="failure-analysis"))


@app.route("/experiment_detail_search", methods=["GET", "POST"])
def experiment_detail_search():
    recent_items, recent_error = api_request("GET", "/experiments")
    recent_experiments = build_recent_experiment_rows(recent_items or [])

    board, board_error = api_request("GET", "/leaderboard")
    board_by_experiment = {}
    for entry in board or []:
        if isinstance(entry, dict) and entry.get("experiment_id"):
            board_by_experiment[str(entry["experiment_id"])] = entry
    experiment_rows = build_experiment_rows(recent_items or [], board_by_experiment)
    recent_runs = build_experiment_run_groups(experiment_rows, limit=2)
    experiment_run_groups = build_experiment_run_groups(experiment_rows)

    registry_context = {
        "experiment_rows": experiment_rows,
        "experiment_run_groups": experiment_run_groups,
        "registry_error": recent_error,
        "leaderboard_error": board_error,
    }

    if request.method == "POST":
        exp_id = (request.form.get("exp_id") or "").strip()
        if not exp_id:
            return render_template(
                "experiment_detail_search.html",
                error="Experiment ID is required.",
                recent_error=recent_error,
                recent_experiments=recent_experiments,
                recent_runs=recent_runs,
                **registry_context,
            )
    else:
        exp_id = (request.args.get("exp_id") or "").strip()

    if exp_id:
        experiment, error = api_request("GET", f"/experiments/{quote(exp_id, safe='')}")
        detail_context = build_experiment_detail_context(experiment)
        return render_template(
            "experiment_detail.html",
            exp_id=exp_id,
            experiment=experiment,
            error=error,
            **detail_context,
        )

    return render_template(
        "experiment_detail_search.html",
        error=None,
        recent_error=recent_error,
        recent_experiments=recent_experiments,
        recent_runs=recent_runs,
        **registry_context,
    )


@app.route("/debug/helpers")
def debug_helpers():
    claim = safe_oauth_call("get_claim", "email")
    organization = safe_oauth_call("get_organization")
    user_organizations = safe_oauth_call("get_user_organizations")
    flag = safe_oauth_call("get_flag", "theme")
    bool_flag = safe_oauth_call("get_boolean_flag", "is_dark_mode")
    str_flag = safe_oauth_call("get_string_flag", "theme")
    int_flag = safe_oauth_call("get_integer_flag", "competitions_limit")

    return render_template(
        "helpers.html",
        claim=serialize_for_template(claim),
        organization=serialize_for_template(organization),
        user_organizations=serialize_for_template(user_organizations),
        flag=serialize_for_template(flag),
        bool_flag=serialize_for_template(bool_flag),
        str_flag=serialize_for_template(str_flag),
        int_flag=serialize_for_template(int_flag),
    )


@app.route("/debug/details")
def debug_details():
    access_token = safe_oauth_call("get_access_token")
    if isinstance(access_token, dict):
        access_token = access_token.get("access_token") or access_token
    return render_template("details.html", access_token=serialize_for_template(access_token))


@app.route("/debug/api-demo")
def debug_api_demo():
    users, error = fetch_management_users()
    return render_template(
        "api_demo.html",
        users=users,
        is_api_call=error is None,
        mgmt_error=error,
    )


if __name__ == "__main__":
    log_dashboard_startup()
    for rule in app.url_map.iter_rules():
        print(rule.endpoint, rule)

    port = int(os.getenv("DASHBOARD_PORT", "8501"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,
        exclude_patterns=["*/retrieval/pipeline.py"],
    )
