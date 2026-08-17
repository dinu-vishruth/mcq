"""Session management: report, PDF/CSV export, and lifecycle actions.

Port of core/routes/teacher.py. Ownership is enforced through one dependency
(``owned_session``) instead of the copy-pasted teacher-check that opened each of
these handlers, so no route can accidentally omit it.

SECURITY NOTE: the Flask version of /download_report checked only that the key
was alphanumeric -- no login, no ownership -- so anyone who guessed an 8-char key
could download that quiz's full question set and answers. This port applies the
same ownership check as its sibling routes. If you need public report links, add
them deliberately (e.g. a signed token) rather than by omission.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from ..deps import CurrentUser, Db
from ..templating import flash, render

router = APIRouter(tags=["sessions"])

#: The route these actions return to after a flash message.
_RETURN_TO = "/dashboard"

_OPTION_LABELS = ("A", "B", "C", "D")
MIN_TIMER_SECONDS, MAX_TIMER_SECONDS = 10, 3600
VALID_DIFFICULTIES = ("easy", "medium", "hard")

#: Session keys are uuid4 prefixes, so anything non-alphanumeric is malformed.
#: Validated in the path itself -- a bad key never reaches a query.
SessionKey = Annotated[str, Path(pattern=r"^[A-Za-z0-9]+$")]


async def owned_session(session_key: SessionKey, user: CurrentUser, conn: Db) -> sqlite3.Row:
    """Fetch a session, or fail with the same 403/404 the Flask version returned.

    ``async`` for the same reason as the ``db`` dependency: a sync dependency runs
    in the threadpool, and a sqlite3 connection may only be used on the thread
    that created it.
    """
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_key=?", (session_key,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if row["teacher"] != user.username:
        raise HTTPException(status_code=403, detail="Access Denied")
    return row


OwnedSession = Annotated[sqlite3.Row, Depends(owned_session)]


@router.get("/teacher")
async def teacher_dashboard():
    """Legacy teacher dashboard removed -- everyone is a single "User" now.

    Kept as a redirect so old links never 404.
    """
    return RedirectResponse(_RETURN_TO, status_code=302)


@router.get("/session_report/{session_key}")
async def session_report(
    request: Request, session_key: SessionKey, session_row: OwnedSession, conn: Db
):
    results = conn.execute(
        """
        SELECT r.*, u.username as registered_name
        FROM results r LEFT JOIN users u ON r.user_id = u.id
        WHERE r.session_key=? ORDER BY r.score DESC
        """,
        (session_key,),
    ).fetchall()
    return render(request, "session_report.html", session_key=session_key, results=results)


@router.get("/download_report/{session_key}")
async def download_report(session_key: SessionKey, session_row: OwnedSession):
    """Render this session's MCQ set as a downloadable PDF."""
    from reportlab.pdfgen import canvas

    os.makedirs("reports", exist_ok=True)
    filename = f"reports/report_{session_key}.pdf"

    pdf = canvas.Canvas(filename)
    y = 800
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, f"MCQ Set - {session_key}")
    y -= 20
    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        40, y,
        f"Teacher: {session_row['teacher']} | Difficulty: {session_row['difficulty']} "
        f"| Timer: {session_row['timer']}s",
    )
    y -= 30

    pdf.setFont("Helvetica", 11)
    for index, question in enumerate(json.loads(session_row["mcqs_json"])):
        # Wrap by hand: reportlab's canvas has no automatic line breaking.
        text = f"Q{index + 1}. {question['question']}"
        max_width = 80
        while len(text) > max_width:
            wrap_at = text.rfind(" ", 0, max_width)
            if wrap_at == -1:
                wrap_at = max_width
            pdf.drawString(40, y, text[:wrap_at])
            text = text[wrap_at:].strip()
            y -= 14
        pdf.drawString(40, y, text)
        y -= 18

        for option in question["options"]:
            pdf.drawString(60, y, f"{option.get('label', '?')}) {option.get('text', '')}")
            y -= 14
        y -= 8

        if y < 80:
            pdf.showPage()
            y = 800
    pdf.save()

    return FileResponse(
        filename, media_type="application/pdf", filename=f"report_{session_key}.pdf"
    )


@router.post("/delete_session/{session_key}")
async def delete_session(
    request: Request, session_key: SessionKey, session_row: OwnedSession, conn: Db
):
    conn.execute("DELETE FROM sessions WHERE session_key=?", (session_key,))
    conn.execute("DELETE FROM results WHERE session_key=?", (session_key,))
    conn.commit()
    flash(request, "Session deleted successfully!", "success")
    return RedirectResponse(_RETURN_TO, status_code=302)


@router.post("/archive_session/{session_key}")
async def archive_session(
    request: Request, session_key: SessionKey, session_row: OwnedSession, conn: Db
):
    conn.execute("UPDATE sessions SET archived=1 WHERE session_key=?", (session_key,))
    conn.commit()
    flash(request, "Session archived successfully!", "success")
    return RedirectResponse(_RETURN_TO, status_code=302)


@router.post("/unarchive_session/{session_key}")
async def unarchive_session(
    request: Request, session_key: SessionKey, session_row: OwnedSession, conn: Db
):
    conn.execute("UPDATE sessions SET archived=0 WHERE session_key=?", (session_key,))
    conn.commit()
    flash(request, "Session unarchived successfully!", "success")
    return RedirectResponse(_RETURN_TO, status_code=302)


@router.post("/clone_session/{session_key}")
async def clone_session(
    request: Request, user: CurrentUser, session_row: OwnedSession, conn: Db
):
    new_key = str(uuid.uuid4())[:8]
    conn.execute(
        "INSERT INTO sessions (session_key, teacher, created_at, difficulty, timer, "
        "mcqs_json, archived) VALUES (?, ?, ?, ?, ?, ?, 0)",
        (new_key, user.username, datetime.utcnow().isoformat(),
         session_row["difficulty"], session_row["timer"], session_row["mcqs_json"]),
    )
    conn.commit()
    flash(request, f"Session cloned successfully! New key: {new_key}", "success")
    return RedirectResponse(_RETURN_TO, status_code=302)


@router.get("/edit_session/{session_key}")
async def edit_session_form(
    request: Request, session_key: SessionKey, session_row: OwnedSession
):
    return render(
        request, "edit_session.html",
        session_key=session_key, timer=session_row["timer"],
        difficulty=session_row["difficulty"], mcqs=json.loads(session_row["mcqs_json"]),
    )


@router.post("/edit_session/{session_key}")
async def edit_session(
    request: Request, session_key: SessionKey, session_row: OwnedSession, conn: Db
):
    form = await request.form()
    original = json.loads(session_row["mcqs_json"])

    try:
        timer = int(form.get("timer") or 60)
        if not MIN_TIMER_SECONDS <= timer <= MAX_TIMER_SECONDS:
            raise ValueError(
                f"Timer must be between {MIN_TIMER_SECONDS} and {MAX_TIMER_SECONDS} seconds."
            )
        difficulty = (form.get("difficulty") or "medium").strip().lower()
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError("Invalid difficulty value.")

        updated = [_parse_edited_question(form, i) for i in range(len(original))]
    except ValueError as exc:
        return render(
            request, "edit_session.html",
            session_key=session_key, timer=session_row["timer"],
            difficulty=session_row["difficulty"], mcqs=original, error=str(exc),
        )

    conn.execute(
        "UPDATE sessions SET timer=?, difficulty=?, mcqs_json=? WHERE session_key=?",
        (timer, difficulty, json.dumps(updated), session_key),
    )
    conn.commit()
    flash(request, "Session updated successfully!", "success")
    return RedirectResponse(_RETURN_TO, status_code=302)


def _parse_edited_question(form: Any, index: int) -> dict[str, Any]:
    """Validate one question from the edit form. Raises ValueError with a message."""
    text = (form.get(f"q_{index}_text") or "").strip()
    if not text:
        raise ValueError(f"Question {index + 1} cannot be empty.")

    options = []
    for label in _OPTION_LABELS:
        option_text = (form.get(f"q_{index}_opt_{label}") or "").strip()
        if not option_text:
            raise ValueError(f"Option {label} of Question {index + 1} cannot be empty.")
        options.append({"label": label, "text": option_text})

    answer = (form.get(f"q_{index}_answer") or "").strip()
    if answer not in [o["text"] for o in options]:
        # Tolerate a bare label ("B") by resolving it to that option's text.
        if answer in _OPTION_LABELS:
            answer = options[_OPTION_LABELS.index(answer)]["text"]
        else:
            raise ValueError(
                f"Correct answer for Question {index + 1} must match one of the option texts."
            )
    return {"question": text, "options": options, "answer_text": answer}


@router.get("/export_results/{session_key}")
async def export_results(session_key: SessionKey, session_row: OwnedSession, conn: Db):
    """Stream this session's results as CSV."""
    results = conn.execute(
        """
        SELECT r.student_name, u.username as registered_name, r.score, r.total,
               r.submitted_at, r.time_spent
        FROM results r LEFT JOIN users u ON r.user_id = u.id
        WHERE r.session_key=? ORDER BY r.score DESC
        """,
        (session_key,),
    ).fetchall()

    def rows():
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        def flush() -> str:
            value = buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
            return value

        writer.writerow([
            "Student Name", "Username", "Score", "Total Questions",
            "Percentage", "Time Spent (s)", "Submitted At",
        ])
        yield flush()

        for row in results:
            percentage = round(row["score"] / row["total"] * 100, 2) if row["total"] else 0
            writer.writerow([
                row["student_name"], row["registered_name"] or "N/A",
                row["score"], row["total"], f"{percentage}%",
                row["time_spent"], row["submitted_at"],
            ])
            yield flush()

    return StreamingResponse(
        rows(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="results_{session_key}.csv"'},
    )
