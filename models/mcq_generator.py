# models/mcq_generator.py
import json
import requests
from config import GROK_API_KEY, GROK_MODEL
from utils.text_cleaner import clean_text

def generate_mcqs(text, num_questions=5, difficulty="medium"):
    """
    Generate high-quality MCQs using xAI Grok API.
    Difficulty: easy / medium / hard
    Returns a list of MCQ dictionaries.
    """
    text = clean_text(text)
    if not text or not GROK_API_KEY:
        print("[WARNING] Missing text or GROK_API_KEY.")
        return []

    # Limit text length to prevent token limit issues
    max_chars = 100000 
    if len(text) > max_chars:
        text = text[:max_chars]

    prompt = f"""You are an expert exam creator. Create {num_questions} Multiple Choice Questions from the text below.
The difficulty must be exactly: {difficulty.upper()}.
Make the distractors highly plausible and challenging (unless difficulty is 'easy').

You MUST respond with a JSON object containing a "questions" key, which holds an array of exactly {num_questions} MCQ objects.
Each MCQ object must have exactly the following structure:
{{
  "question": "The question text",
  "options": [
    {{"label": "A", "text": "Option A text"}},
    {{"label": "B", "text": "Option B text"}},
    {{"label": "C", "text": "Option C text"}},
    {{"label": "D", "text": "Option D text"}}
  ],
  "answer_text": "The exact text of the correct option (must match one of the options' text)"
}}

TEXT:
\"\"\"
{text}
\"\"\"
"""

    # Auto-detect if using Groq (starts with gsk_) or xAI (Grok)
    if GROK_API_KEY.startswith("gsk_"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        model = GROK_MODEL if "llama" in GROK_MODEL.lower() else "llama-3.3-70b-versatile"
    else:
        url = "https://api.xai.com/v1/chat/completions"
        model = GROK_MODEL

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert exam creator that only outputs valid JSON objects."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=45
        )
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        
        if "questions" in data:
            mcqs = data["questions"]
        elif isinstance(data, list):
            mcqs = data
        else:
            mcqs = []
            
        # Validate structure to prevent errors
        valid_mcqs = []
        for q in mcqs:
            if "question" in q and "options" in q and "answer_text" in q:
                if len(q["options"]) == 4:
                    options_valid = True
                    for o in q["options"]:
                        if not isinstance(o, dict) or "label" not in o or "text" not in o:
                            options_valid = False
                            break
                    if options_valid:
                        valid_mcqs.append(q)
                        
        return valid_mcqs[:num_questions]
        
    except Exception as e:
        print(f"[ERROR] Failed to generate MCQs with Grok API: {e}")
        return []
