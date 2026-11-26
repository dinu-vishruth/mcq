"""Verified owner dashboard and CSRF-protected management actions."""
import time
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from core.services.admin_control import CONTROLS, is_owner, update_control, update_user_access

from ..deps import CurrentUser, Db, User
from ..templating import flash, render, templates

router = APIRouter(prefix="/admin", tags=["owner"])


async def require_owner(user: CurrentUser, conn: Db) -> User:
    row = conn.execute("SELECT * FROM users WHERE id=?", (user.id,)).fetchone()
    if not is_owner(row):
        raise HTTPException(403, "This dashboard is reserved for the verified application owner.")
    return user


Owner = Annotated[User, Depends(require_owner)]


def _time_label(value):
    if value is None:
        return "Not recorded"
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    return str(value)[:19].replace("T", " ") + " UTC"


templates.env.filters["admin_time"] = _time_label


@router.get("")
async def dashboard(request: Request, user: Owner, conn: Db,
                    q: Annotated[str, Query(max_length=100)] = "",
                    page: Annotated[int, Query(ge=1, le=100000)] = 1):
    now = time.time()
    stats = dict(conn.execute(
        "SELECT count(*) AS accounts, sum(CASE WHEN auth_provider='demo' THEN 1 ELSE 0 END) AS demos, "
        "sum(CASE WHEN is_disabled=1 THEN 1 ELSE 0 END) AS disabled, "
        "sum(CASE WHEN is_disabled=0 AND last_seen_at>? THEN 1 ELSE 0 END) AS recent FROM users",
        (now - 900,),
    ).fetchone())
    stats["sign_ins"] = conn.execute("SELECT count(*) FROM access_events WHERE event='sign_in' AND occurred_at>?", (now - 86400,)).fetchone()[0]
    stats["jobs"] = conn.execute("SELECT count(*) FROM generation_jobs WHERE status IN ('queued','running')").fetchone()[0]
    search = q.strip().replace("!", "!!").replace("%", "!%").replace("_", "!_").lower()
    where = " WHERE lower(username) LIKE ? ESCAPE '!' OR lower(COALESCE(email,'')) LIKE ? ESCAPE '!'" if search else ""
    params = (f"%{search}%", f"%{search}%") if search else ()
    total = conn.execute("SELECT count(*) FROM users" + where, params).fetchone()[0]
    pages = max(1, (total + 19) // 20)
    page = min(page, pages)
    users = [dict(row) for row in conn.execute(
        "SELECT id,username,email,email_verified,auth_provider,created_at,is_disabled,last_seen_at FROM users"
        + where + " ORDER BY id DESC LIMIT 20 OFFSET ?", (*params, (page - 1) * 20)).fetchall()]
    for account in users:
        account["is_owner"] = is_owner(account)
    events = conn.execute("SELECT username,event,ip,occurred_at FROM access_events WHERE occurred_at>? ORDER BY id DESC LIMIT 20",
                          (now - 30 * 86400,)).fetchall()
    failures = conn.execute("SELECT username,ip,attempt_time FROM login_attempts WHERE success=0 ORDER BY id DESC LIMIT 10").fetchall()
    audit = conn.execute("SELECT actor_email,action,target,occurred_at FROM admin_audit ORDER BY id DESC LIMIT 20").fetchall()
    states = {row["name"]: bool(row["enabled"]) for row in conn.execute("SELECT name,enabled FROM app_controls").fetchall()}
    return render(request, "admin.html", owner=user, stats=stats, accounts=users, events=events,
                  failures=failures, audit=audit, controls=CONTROLS, states=states, q=q, page=page,
                  pages=pages, total=total, can_manage=now - request.session.get("signed_in_at", 0) <= 900)


def _recent_sign_in(request):
    if time.time() - request.session.get("signed_in_at", 0) > 900:
        raise HTTPException(403, "Sign out and sign back in before making admin changes.")


@router.post("/controls/{name}")
async def control(request: Request, name: str, user: Owner, conn: Db,
                  value: Annotated[int, Form(ge=0, le=1)]):
    _recent_sign_in(request)
    try:
        update_control(conn, user.id, name, bool(value))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except PermissionError:
        raise HTTPException(403, "Owner access is required.") from None
    flash(request, "Application control updated.", "success")
    return RedirectResponse("/admin#controls", status_code=303)


@router.post("/accounts/{target_id}/{action}")
async def account_action(request: Request, target_id: int, action: str, user: Owner, conn: Db):
    _recent_sign_in(request)
    try:
        update_user_access(conn, user.id, target_id, action)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except PermissionError:
        raise HTTPException(403, "Owner access is required.") from None
    flash(request, "Account access updated.", "success")
    return RedirectResponse("/admin#accounts", status_code=303)
