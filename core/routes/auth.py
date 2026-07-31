"""
Auth + account blueprint: home, signup, login (with lockout), logout, profile,
delete_account. Behaviour is a verbatim move from app.py; only url_for targets
are blueprint-namespaced (auth.home, teacher.teacher_dashboard,
student.student_dashboard). Route paths are unchanged, so templates that use
hardcoded paths keep working.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from flask import (Blueprint, render_template, request, session, redirect,
                   url_for, flash)
from werkzeug.security import generate_password_hash, check_password_hash

from core.models.db import get_db

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("student.dashboard"))
    return render_template("login.html")


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not username or not password:
        return render_template("signup.html", error="All fields are required.")

    if len(password) < 8:
        return render_template("signup.html", error="Password must be at least 8 characters long.")

    hashed_pw = generate_password_hash(password)

    # Single unified "User" role. The DB role column has a legacy CHECK constraint
    # (teacher|student); we write 'student' to satisfy it but never branch on it.
    conn = get_db()
    try:
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                     (username, hashed_pw, "student"))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return render_template("signup.html", error="Username already exists.")
    conn.close()

    return redirect(url_for("auth.home"))


@auth_bp.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    ip = request.remote_addr

    conn = get_db()

    five_mins_ago = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
    attempts = conn.execute("""
        SELECT COUNT(*) FROM login_attempts
        WHERE (ip=? OR username=?) AND success=0 AND attempt_time > ?
    """, (ip, username, five_mins_ago)).fetchone()[0]

    if attempts >= 5:
        conn.close()
        return render_template("login.html", error="Too many failed login attempts. Please wait 5 minutes.")

    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

    if user and check_password_hash(user["password_hash"], password):
        conn.execute("INSERT INTO login_attempts (ip, username, attempt_time, success) VALUES (?, ?, ?, 1)",
                     (ip, username, datetime.utcnow().isoformat()))
        conn.commit()

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        conn.close()

        return redirect(url_for("student.dashboard"))

    conn.execute("INSERT INTO login_attempts (ip, username, attempt_time, success) VALUES (?, ?, ?, 0)",
                 (ip, username, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    return render_template("login.html", error="Invalid username or password")


@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.home"))


@auth_bp.route("/profile", methods=["GET", "POST"])
def profile():
    if not session.get("user_id"):
        return redirect(url_for("auth.home"))

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (session.get("user_id"),)).fetchone()

    if request.method == "GET":
        conn.close()
        return render_template("profile.html", user=user)

    try:
        new_username = request.form.get("username", "").strip()
        new_email = request.form.get("email", "").strip()
        new_password = request.form.get("password", "").strip()

        if not new_username:
            raise ValueError("Username is required.")

        if new_username != user["username"]:
            chk = conn.execute("SELECT 1 FROM users WHERE username=?", (new_username,)).fetchone()
            if chk:
                raise ValueError("Username already exists.")

        if new_password and len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters long.")

        if new_password:
            hashed_pw = generate_password_hash(new_password)
            conn.execute("UPDATE users SET username=?, email=?, password_hash=? WHERE id=?",
                         (new_username, new_email, hashed_pw, session.get("user_id")))
        else:
            conn.execute("UPDATE users SET username=?, email=? WHERE id=?",
                         (new_username, new_email, session.get("user_id")))
        conn.commit()
        session["username"] = new_username
        conn.close()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("auth.home"))

    except ValueError as e:
        conn.close()
        return render_template("profile.html", user=user, error=str(e))


@auth_bp.route("/delete_account", methods=["POST"])
def delete_account():
    user_id = session.get("user_id")
    username = session.get("username")
    if not user_id:
        return redirect(url_for("auth.home"))

    # A single user may own both generated sessions (by username) and quiz
    # results (by user_id), plus learning data. Remove all of it.
    conn = get_db()
    conn.execute("DELETE FROM results WHERE session_key IN (SELECT session_key FROM sessions WHERE teacher=?)", (username,))
    conn.execute("DELETE FROM sessions WHERE teacher=?", (username,))
    conn.execute("DELETE FROM results WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM learning_history WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM weak_topics WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM user_prefs WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    session.clear()
    flash("Your account has been deleted.", "success")
    return redirect(url_for("auth.home"))
