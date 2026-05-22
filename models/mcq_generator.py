# models/mcq_generator.py
import json
import requests
from config import GROK_API_KEY, GROK_MODEL
from utils.text_cleaner import clean_text

class MCQGenerationError(Exception):
    """Custom exception raised when MCQ generation fails."""
    pass

def generate_mcqs(text, num_questions=5, difficulty="medium"):
    """
    Generate high-quality MCQs using xAI Grok API.
    Difficulty: easy / medium / hard
    Returns a list of MCQ dictionaries.
    """
    text = clean_text(text)
    if not GROK_API_KEY:
        raise MCQGenerationError("API Key is missing. Please set the GROK_API_KEY environment variable in your .env file.")
    if not text:
        raise MCQGenerationError("No text found. Please upload a valid document or provide some text.")

    # Limit text length to prevent token limit issues
    max_chars = 150000 
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
        if response.status_code == 401:
            raise MCQGenerationError("Invalid API key! Please check the GROK_API_KEY in your .env file.")
        elif response.status_code == 429:
            raise MCQGenerationError("API rate limit exceeded! Please wait a moment and try again.")
        elif response.status_code != 200:
            raise MCQGenerationError(f"API request failed with status code {response.status_code}: {response.text}")
            
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
    except requests.exceptions.Timeout:
        raise MCQGenerationError("The request to the AI API timed out. Please try again with a shorter text.")
    except requests.exceptions.RequestException as re:
        raise MCQGenerationError(f"Network error when connecting to the AI API: {str(re)}")
    except Exception as e:
        if not isinstance(e, MCQGenerationError):
            raise MCQGenerationError(f"Unexpected error: {str(e)}")
        raise e

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise MCQGenerationError("The AI service returned an invalid JSON response. Please try again.")

    if "questions" in data:
        mcqs = data["questions"]
    elif isinstance(data, list):
        mcqs = data
    else:
        raise MCQGenerationError("JSON structure is missing 'questions' array key.")
        
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
                    
    if not valid_mcqs:
        raise MCQGenerationError("No valid questions could be structured from the AI response. Please retry.")

    return valid_mcqs[:num_questions]
