"""Shared rules for account names, contact addresses, and sign-in sessions."""
import re
import sqlite3
import unicodedata

SESSION_VERSION = 2
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
            and row["auth_provider"] not in ("google", "github", "demo"))


def establish_session(request, user) -> None:
    request.session.clear()
    request.session.update(user_id=user["id"], username=user["username"], identity_version=SESSION_VERSION)


def rename_username_owned_rows(conn, old: str, new: str) -> None:
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
