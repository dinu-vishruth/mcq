# config.py
import os

SECRET_KEY = os.getenv("SECRET_KEY", "replace_this_with_secure_random_string")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DB_PATH = "/tmp/mcq.db" if os.environ.get("VERCEL") else "database/mcq.db"
UPLOAD_FOLDER = "/tmp/uploads" if os.environ.get("VERCEL") else "uploads"
ALLOWED_EXT = {"pdf", "docx", "txt", "pptx"}
