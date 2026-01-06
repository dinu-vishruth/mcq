# utils/text_cleaner.py
import re

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    # remove strange control chars
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    return text.strip()
