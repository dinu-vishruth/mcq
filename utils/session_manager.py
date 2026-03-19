# utils/session_manager.py
import uuid
import sqlite3
import json
from datetime import datetime
from config import DB_PATH

def create_session_key(teacher, difficulty, timer, mcqs):
    key = str(uuid.uuid4())[:8]
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (session_key, teacher, created_at, difficulty, timer, mcqs_json) VALUES (?, ?, ?, ?, ?, ?)",
                (key, teacher, datetime.utcnow().isoformat(), difficulty, timer, json.dumps(mcqs)))
    conn.commit()
    conn.close()
    return key

def validate_session_key(key):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sessions WHERE session_key=?", (key,))
    res = cur.fetchone()
    conn.close()
    return res is not None
