"""Shared rules for account names, contact addresses, and sign-in sessions."""
import re
import sqlite3
import time
import unicodedata

SESSION_VERSION = 3
_EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def normalize_email(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if len(value) > 254 or not _EMAIL.fullmatch(value):
        raise ValueError("Enter a valid email address.")
    return value.lower()


def normalize_username(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Username is required.")
    if "@" in unicodedata.normalize("NFKC", value):
        raise ValueError("Choose a username without an email address or @ symbol.")
    if any(c in value for c in ("/", "\\")) or value in (".", ".."):
        raise ValueError("Choose a username without path separators.")
    if len(value) > 64 or any(not char.isprintable() for char in value):
        raise ValueError("Choose a username with at most 64 printable characters.")
    return value


def identity_write_lock(conn) -> None:
    """Serialize claims with the account write that follows in this transaction."""
    if isinstance(conn, sqlite3.Connection):
        conn.execute("BEGIN IMMEDIATE")
    else:
        conn.execute("SELECT pg_advisory_xact_lock(7842302)")


def username_exists(conn, username: str, *, exclude_user_id: int | None = None) -> bool:
    sql = "SELECT 1 FROM users WHERE lower(username)=lower(?)"
    params = (username,)
    if exclude_user_id is not None:
        sql += " AND id<>?"
        params += (exclude_user_id,)
    return (conn.execute(sql, params).fetchone() is not None
            or conn.execute("SELECT 1 FROM retired_usernames WHERE name=lower(?)", (username,)).fetchone() is not None)


def unique_oauth_username(conn, *names: str) -> str:
    for raw in names:
        if not isinstance(raw, str) or "@" in unicodedata.normalize("NFKC", raw):
            continue
        name = re.sub(r"[^a-zA-Z0-9 _.-]", "", raw).strip(" _.-")[:32]
        if name:
            break
    else:
        name = "Learner"
    candidate, suffix = name, 2
    while username_exists(conn, candidate):
        candidate = f"{name}{suffix}"
        suffix += 1
    return candidate


def claim_verified_email(conn, email: str, *, user_id: int | None = None) -> bool:
    """Reserve a provider-verified address without linking the old account."""
    existing = conn.execute(
        "SELECT id, email, email_verified FROM users WHERE lower(trim(email))=?", (email,),
    ).fetchone()
    if existing is None or existing["id"] == user_id:
        return True
    if existing["email_verified"]:
        return False
    conn.execute("INSERT INTO identity_email_conflicts (user_id, former_email) VALUES (?, ?) "
                 "ON CONFLICT (user_id, former_email) DO NOTHING", (existing["id"], existing["email"]))
    conn.execute("UPDATE users SET email=NULL, email_verified=0 WHERE id=?", (existing["id"],))
    return True


def password_signin_allowed(row) -> bool:
    return (not row["google_id"] and not row["github_id"]
            and row["auth_provider"] not in ("google", "github", "demo")
            and not (row["is_disabled"] if "is_disabled" in row.keys() else False))


def establish_session(request, user, conn=None) -> None:
    if "is_disabled" in user.keys() and user["is_disabled"]:
        from fastapi import HTTPException
        raise HTTPException(403, "This account has been disabled by the site owner.")
    request.session.clear()
    epoch = user["session_epoch"] if "session_epoch" in user.keys() else 0
    request.session.update(user_id=user["id"], username=user["username"], identity_version=SESSION_VERSION,
                           session_epoch=epoch, signed_in_at=time.time())
    if conn is not None:
        from core.services.admin_control import record_access
        provider = user["auth_provider"] if "auth_provider" in user.keys() else None
        record_access(conn, request, user, "demo" if provider == "demo" else "sign_in")


def link_provider(conn, request, provider, subject, email):
    target = request.session.pop(f"_{provider}_link_target", None)
    if not target:
        return None
    if target["expires"] < time.time() or target["user_id"] != request.session.get("user_id"):
        raise ValueError("Account linking expired. Sign in again.")
    identity_write_lock(conn)
    row = conn.execute("SELECT * FROM users WHERE id=?", (target["user_id"],)).fetchone()
    column = "google_id" if provider == "google" else "github_id"
    other = conn.execute(f"SELECT id FROM users WHERE {column}=?", (subject,)).fetchone()
    if row is None or (other and other["id"] != row["id"]) or (row[column] and row[column] != subject):
        raise ValueError("This provider already belongs to another account. Accounts were not merged.")
    if email:
        email = normalize_email(email)
        if row["email"] and normalize_email(row["email"]) != email:
            raise ValueError("Use a provider account with the same verified email.")
        if not claim_verified_email(conn, email, user_id=row["id"]):
            raise ValueError("This email belongs to another account.")
        conn.execute("UPDATE users SET email=?, email_verified=1 WHERE id=?", (email, row["id"]))
    conn.execute(f"UPDATE users SET {column}=?, password_hash='!', session_epoch=session_epoch+1 WHERE id=?", (subject, row["id"]))
    conn.commit()
    return conn.execute("SELECT * FROM users WHERE id=?", (row["id"],)).fetchone()


def rename_username_owned_rows(conn, old: str, new: str) -> None:
    conn.execute("INSERT INTO upload_prefixes(user_id,prefix) SELECT id,? FROM users WHERE username=? "
                 "ON CONFLICT(user_id,prefix) DO NOTHING", (old + "/", new))
    if old.lower() != new.lower():
        # In-flight requests can still hold the old name. Keep it out of use
        # so their work cannot be attributed to another account.
        conn.execute("INSERT INTO retired_usernames (name) VALUES (lower(?)) ON CONFLICT (name) DO NOTHING", (old,))
    for table, field in (
        ("documents", "owner"), ("sessions", "teacher"), ("generation_jobs", "username"),
        ("generation_admissions", "username"), ("login_attempts", "username"),
    ):
        conn.execute(f"UPDATE {table} SET {field}=? WHERE {field}=?", (new, old))
    conn.execute("UPDATE generation_leases SET scope=? WHERE scope=?", (f"user:{new}", f"user:{old}"))
