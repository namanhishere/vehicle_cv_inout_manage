"""Flask admin web UI: login-protected LAN dashboard.

- Session: Flask signed cookies, secret loaded/created at secret_file
  (mode 600, 32 random bytes); HttpOnly + SameSite=Lax.
- Auth: scrypt hash at password_file (gate.cli passwd writes it);
  3 failed attempts -> 5 s per-IP lockout.
- CSRF: per-session token issued on GET /login, hidden field on every
  form, verified on every POST (400 on mismatch/missing).
- No debug endpoints; app.debug=False always. Crop images are never
  served. All output Jinja-escaped by default.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from gate.plate import normalize

log = logging.getLogger("gate.web")

_PASSWORD_SCHEME = "scrypt"
_LOCKOUT_MAX = 3
_LOCKOUT_S = 5.0

# reason -> dashboard label for the last-event line
_REASON_LABEL = {
    "ALLOW": "registered",
    "UNREGISTERED": "unregistered",
    "LOW_CONF": "low confidence",
    "INVALID_FORMAT": "invalid",
    "ALREADY_INSIDE": "registered",
    "ALREADY_OUTSIDE": "registered",
}


def _load_or_create_secret(path: str) -> bytes:
    """32 random bytes at path (mode 600); create when missing."""
    if os.path.exists(path):
        with open(path, "rb") as fh:
            data = fh.read()
        if len(data) >= 16:
            return data
    secret = secrets.token_bytes(32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, secret)
    finally:
        os.close(fd)
    return secret


def verify_password(password: str, stored: str) -> bool:
    """scrypt$<n>$<r>$<p>$<salt_hex>$<hash_hex> comparison."""
    parts = stored.strip().split("$")
    if len(parts) != 6 or parts[0] != _PASSWORD_SCHEME:
        return False
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt = bytes.fromhex(parts[4])
        expected = bytes.fromhex(parts[5])
    except ValueError:
        return False
    try:
        actual = hashlib_scrypt(password, salt=salt, n=n, r=r, p=p)
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected)


def hashlib_scrypt(password, salt, n, r, p):
    import hashlib

    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p)


class _LoginGuard:
    """Per-IP failure lockout (in-memory)."""

    def __init__(self):
        self._fails: dict[str, tuple[int, float]] = {}

    def blocked(self, ip: str) -> bool:
        import time

        rec = self._fails.get(ip)
        if rec is None:
            return False
        count, until = rec
        if time.time() >= until:
            del self._fails[ip]
            return False
        return count >= _LOCKOUT_MAX

    def fail(self, ip: str) -> None:
        import time

        now = time.time()
        count, until = self._fails.get(ip, (0, 0.0))
        count += 1
        self._fails[ip] = (count, now + _LOCKOUT_S)

    def ok(self, ip: str) -> None:
        self._fails.pop(ip, None)


def create_app(db, config, state) -> Flask:
    """App factory. ``state`` exposes camera_ok(side) and lives in gate_app."""
    app = Flask(__name__)
    app.debug = False
    app.config.update(
        SECRET_KEY=_load_or_create_secret(config.web.secret_file),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        MAX_CONTENT_LENGTH=16 * 1024,
    )
    guard = _LoginGuard()
    _password_file = config.web.password_file

    def _read_password_hash() -> str:
        try:
            with open(_password_file, "r", encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            return ""

    def _logged_in() -> bool:
        return bool(session.get("auth"))

    @app.context_processor
    def _inject():
        return {"csrf_token": lambda: session.get("csrf", "")}

    @app.before_request
    def _csrf():
        if request.method == "POST" and request.form.get("_csrf") != session.get(
            "csrf"
        ):
            abort(400, "CSRF token missing or invalid")

    def login_required(view):
        from functools import wraps

        @wraps(view)
        def wrapped(*args, **kwargs):
            if not _logged_in():
                return redirect(url_for("login", next=request.path))
            return view(*args, **kwargs)

        return wrapped

    # -- auth --------------------------------------------------------------

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            session["csrf"] = secrets.token_hex(16)
            return render_template("login.html")
        ip = request.remote_addr or "?"
        if guard.blocked(ip):
            return render_template(
                "login.html", error="Too many attempts - try again in a few seconds"
            ), 429
        stored = _read_password_hash()
        password = request.form.get("password", "")
        if stored and verify_password(password, stored):
            guard.ok(ip)
            session["auth"] = True
            return redirect(request.args.get("next") or url_for("dashboard"))
        guard.fail(ip)
        return render_template("login.html", error="Wrong password"), 401

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # -- dashboard ---------------------------------------------------------

    @app.route("/")
    @login_required
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/api/status")
    @login_required
    def api_status():
        last = db.last_event()
        last_event = None
        if last is not None:
            last_event = {
                "plate": last["plate"] or "(none)",
                "direction": last["direction"],
                "result": last["result"],
                "reason": _REASON_LABEL.get(last["reason"], last["reason"]),
                "ts": last["ts"],
            }
        return jsonify(
            {
                "system": "ONLINE" if db.ping() else "OFFLINE",
                "cameras": {
                    "IN": bool(state.camera_ok("IN")),
                    "OUT": bool(state.camera_ok("OUT")),
                },
                "database": db.ping(),
                "inside_count": db.inside_count(),
                "last_event": last_event,
            }
        )

    # -- vehicles ----------------------------------------------------------

    @app.route("/vehicles", methods=["GET"])
    @login_required
    def vehicles():
        return render_template("vehicles.html", rows=db.list_vehicles())

    @app.route("/vehicles/add", methods=["POST"])
    @login_required
    def vehicles_add():
        raw = request.form.get("plate", "").strip()
        note = request.form.get("note", "").strip()
        parsed = normalize(raw)
        if parsed is None:
            flash(f"Invalid plate format: {raw!r}", "error")
            return redirect(url_for("vehicles")), 400
        try:
            db.add_vehicle(parsed.canonical, note=note)
        except ValueError:
            flash(f"Vehicle already registered: {parsed.canonical}", "error")
            return redirect(url_for("vehicles"))
        flash(f"Registered {parsed.canonical}", "ok")
        return redirect(url_for("vehicles"))

    @app.route("/vehicles/<plate>/toggle", methods=["POST"])
    @login_required
    def vehicles_toggle(plate):
        row = db.lookup(plate)
        if row is None:
            abort(404)
        db.set_registered(plate, not row["registered"])
        flash(f"{plate}: {'unregistered' if row['registered'] else 'registered'}")
        return redirect(url_for("vehicles"))

    @app.route("/vehicles/<plate>/remove", methods=["POST"])
    @login_required
    def vehicles_remove(plate):
        if db.lookup(plate) is None:
            abort(404)
        db.remove_vehicle(plate)
        flash(f"Removed {plate}")
        return redirect(url_for("vehicles"))

    # -- events ------------------------------------------------------------

    @app.route("/events")
    @login_required
    def events():
        try:
            page = max(1, int(request.args.get("page", "1")))
        except ValueError:
            page = 1
        page = min(page, 1 << 30)
        rows, total = db.list_events(page=page, per_page=25)
        pages = max(1, (total + 24) // 25)
        return render_template(
            "events.html", rows=rows, page=page, pages=pages, total=total
        )

    return app
