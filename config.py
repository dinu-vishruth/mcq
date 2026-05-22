# config.py
import os
import secrets
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-1212")
DB_PATH = "/tmp/mcq.db" if os.environ.get("VERCEL") else "database/mcq.db"
UPLOAD_FOLDER = "/tmp/uploads" if os.environ.get("VERCEL") else "uploads"
ALLOWED_EXT = {"pdf", "docx", "txt", "pptx"}
