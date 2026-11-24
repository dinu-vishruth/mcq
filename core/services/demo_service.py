"""Create a private demo workspace without asking for credentials."""
import secrets
from uuid import uuid4

from werkzeug.security import generate_password_hash

SAMPLE_TITLE = "Aptitude Basics - Demo Material"
SAMPLE_TEXT = """
Percentages express a quantity as a fraction of 100. To find 20 percent of 250,
multiply 250 by 20 and divide by 100, giving 50. A percentage increase equals
the increase divided by the original value, multiplied by 100.
Profit equals selling price minus cost price. Profit percentage uses cost price
as the denominator. An item bought for 80 and sold for 100 yields a profit of
20 and a profit percentage of 25 percent. Loss equals cost price minus selling
price. An item bought for 100 and sold for 90 gives a loss of 10 percent.
Simple interest equals principal times annual rate times years, divided by 100.
For a principal of 1000 at 5 percent per year for 2 years, simple interest is
100 and the final amount is 1100. Simple interest does not earn interest itself.
Speed equals distance divided by time. A journey of 120 kilometres completed
in 2 hours has an average speed of 60 kilometres per hour. For a fixed distance,
doubling the speed halves the travel time. Distance equals speed times time.
An average equals the sum of the values divided by their count. The average
of 10, 20 and 30 is 20. A ratio compares quantities by division. Dividing 50 in
the ratio 2 to 3 gives 20 and 30 because there are 5 equal parts in total.
A rectangle's area equals length times width. Its perimeter equals twice the
sum of length and width. A rectangle 8 metres long and 3 metres wide has an
area of 24 square metres and a perimeter of 22 metres. A triangle's interior
angles sum to 180 degrees. A right angle measures 90 degrees.
""".strip()


def create_demo_workspace(conn) -> dict:
    from core.services.ingestion_service import ingest_document

    username = "Demo Learner " + uuid4().hex[:12]
    row = conn.execute(
        "INSERT INTO users (username, password_hash, role, auth_provider) "
        "VALUES (?, ?, 'user', 'demo') RETURNING id",
        (username, generate_password_hash(secrets.token_urlsafe(32))),
    ).fetchone()
    user_id = row["id"]
    conn.execute(
        "INSERT INTO user_prefs (user_id, goal, style, daily_minutes, onboarded) "
        "VALUES (?, 'placement', 'practice', 30, 1)", (user_id,),
    )
    conn.commit()
    try:
        ingest_document(SAMPLE_TEXT, owner=username, title=SAMPLE_TITLE, source_type="paste")
    except Exception:
        # Do not leave an unusable account behind if preparation fails.
        from core.repositories import document_repo
        for document in document_repo.list_by_owner(username):
            document_repo.delete(document["id"], username)
        conn.execute("DELETE FROM user_prefs WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        raise
    return {"id": user_id, "username": username}
