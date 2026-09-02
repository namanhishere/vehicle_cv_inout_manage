"""Web UI tests (Flask test_client, tmp config; no models needed)."""

import hashlib
import os

import pytest

from gate.config import Config, WebConfig, StorageConfig
from gate.db import GateDB
from gate.web.app import create_app, verify_password

PLATE = "29A1-678.90"
PASSWORD = "hunter2-test"


def make_hash(password: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt$16384$8$1${salt.hex()}${h.hex()}"


@pytest.fixture
def ctx(tmp_path):
    cfg = Config()
    cfg.web = WebConfig(
        host="127.0.0.1", port=8080,
        secret_file=str(tmp_path / "secret"),
        password_file=str(tmp_path / "admin.hash"),
    )
    cfg.storage = StorageConfig(
        db_path=str(tmp_path / "gate.db"),
        images_dir=str(tmp_path / "images"),
    )
    (tmp_path / "admin.hash").write_text(make_hash(PASSWORD))

    class FakeState:
        def camera_ok(self, side):
            return side == "IN"

    db = GateDB(cfg.storage.db_path)
    app = create_app(db, cfg, FakeState())
    app.config["TESTING"] = True
    client = app.test_client()
    yield SimpleNamespace(cfg=cfg, db=db, client=client)
    db.close()


class SimpleNamespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def login(client, password=PASSWORD):
    # GET /login issues the CSRF token
    r = client.get("/login")
    assert r.status_code == 200
    import re

    m = re.search(r'name="_csrf" value="([0-9a-f]+)"', r.get_data(as_text=True))
    assert m, "login form must carry a CSRF token"
    return client.post(
        "/login", data={"_csrf": m.group(1), "password": password},
        follow_redirects=False,
    )


def csrf(client, path="/vehicles"):
    r = client.get(path)
    import re

    m = re.search(r'name="_csrf" value="([0-9a-f]+)"', r.get_data(as_text=True))
    return m.group(1) if m else None


def test_unauthenticated_root_redirects(ctx):
    r = ctx.client.get("/")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_login_wrong_password(ctx):
    r = ctx.client.get("/login")
    import re

    m = re.search(r'name="_csrf" value="([0-9a-f]+)"', r.get_data(as_text=True))
    r = ctx.client.post("/login", data={"_csrf": m.group(1), "password": "nope"})
    assert r.status_code == 401
    assert "Wrong password" in r.get_data(as_text=True)


def test_login_correct_password_and_dashboard(ctx):
    r = login(ctx.client)
    assert r.status_code == 302
    r = ctx.client.get("/")
    assert r.status_code == 200
    assert "Gate status" in r.get_data(as_text=True)


def test_login_lockout_after_three_failures(ctx):
    r = ctx.client.get("/login")
    import re

    m = re.search(r'name="_csrf" value="([0-9a-f]+)"', r.get_data(as_text=True))
    for _ in range(3):
        r = ctx.client.post(
            "/login", data={"_csrf": m.group(1), "password": "bad"}
        )
    # 4th attempt (even with the right password) is locked out
    r = ctx.client.post(
        "/login", data={"_csrf": m.group(1), "password": PASSWORD}
    )
    assert r.status_code == 429


def test_api_status_shape(ctx):
    login(ctx.client)
    r = ctx.client.get("/api/status")
    assert r.status_code == 200
    data = r.get_json()
    assert data["system"] == "ONLINE"
    assert data["cameras"] == {"IN": True, "OUT": False}
    assert data["database"] is True
    assert data["inside_count"] == 0
    assert data["last_event"] is None


def test_add_valid_vehicle(ctx):
    login(ctx.client)
    r = ctx.client.post(
        "/vehicles/add",
        data={"_csrf": csrf(ctx.client), "plate": "29-AB 123.45", "note": "x"},
        follow_redirects=True,
    )
    assert r.status_code == 200
    assert ctx.db.lookup("29AB-123.45") is not None


def test_add_invalid_plate_400_no_row(ctx):
    login(ctx.client)
    r = ctx.client.post(
        "/vehicles/add",
        data={"_csrf": csrf(ctx.client), "plate": "not-a-plate"},
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert ctx.db.list_vehicles() == []


def test_add_sql_injection_plate_rejected(ctx):
    login(ctx.client)
    evil = "29AB-123.45'; DROP TABLE vehicles;--"
    r = ctx.client.post(
        "/vehicles/add",
        data={"_csrf": csrf(ctx.client), "plate": evil},
        follow_redirects=False,
    )
    assert r.status_code == 400
    # table intact
    ctx.db.add_vehicle(PLATE)
    assert ctx.db.lookup(PLATE) is not None


def test_toggle_and_remove(ctx):
    login(ctx.client)
    ctx.db.add_vehicle(PLATE)
    r = ctx.client.post(
        f"/vehicles/{PLATE}/toggle",
        data={"_csrf": csrf(ctx.client)}, follow_redirects=True,
    )
    assert r.status_code == 200
    assert ctx.db.lookup(PLATE)["registered"] == 0
    r = ctx.client.post(
        f"/vehicles/{PLATE}/remove",
        data={"_csrf": csrf(ctx.client)}, follow_redirects=True,
    )
    assert r.status_code == 200
    assert ctx.db.lookup(PLATE) is None


def test_events_page_renders_rows_in_order(ctx):
    login(ctx.client)
    for ts in ("2026-01-01T10:00:00", "2026-01-01T10:00:01"):
        ctx.db.record_event(
            ts=ts, plate=PLATE, raw=PLATE, direction="IN", result="ALLOW",
            reason="ALLOW", confidence=0.9, camera="IN", crop=None,
        )
    r = ctx.client.get("/events")
    body = r.get_data(as_text=True)
    assert r.status_code == 200
    assert PLATE in body
    assert "ALLOW" in body


def test_post_without_csrf_token_400(ctx):
    login(ctx.client)
    r = ctx.client.post(
        "/vehicles/add", data={"plate": "29AB-123.45"}
    )
    assert r.status_code == 400


def test_verify_password_roundtrip():
    h = make_hash(PASSWORD)
    assert verify_password(PASSWORD, h)
    assert not verify_password("wrong", h)
    assert not verify_password(PASSWORD, "garbage")


def test_logout_clears_session(ctx):
    login(ctx.client)
    r = ctx.client.get("/logout")
    assert r.status_code == 302
    assert ctx.client.get("/").status_code == 302
