# models/mcq_generator.py
import json
import re
import google.generativeai as genai
from config import GEMINI_API_KEY
from utils.text_cleaner import clean_text

# Configure Gemini AI
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def generate_mcqs(text, num_questions=5, difficulty="medium"):
    """
    Generate high-quality MCQs using Google Gemini AI.
    Difficulty: easy / medium / hard
    Returns a list of MCQ dictionaries.
    """
    text = clean_text(text)
    if not text or not GEMINI_API_KEY:
        return []

    # Trim text to prevent token limit overflows for massive PDFs
    # Gemini 1.5 Flash has a very large context window, but we limit for speed/safety
    max_chars = 100000 
    if len(text) > max_chars:
        text = text[:max_chars]

    prompt = f"""
    You are an expert exam creator. create {num_questions} Multiple Choice Questions from the text below.
    The difficulty must be exactly: {difficulty.upper()}.
    Make the distractors highly plausible and challenging (unless difficulty is 'easy').
    
    You MUST respond with a perfectly valid JSON array containing exactly {num_questions} objects in the following format:
    [
      {{
        "question": "What is...",
        "options": [
          {{"label": "A", "text": "Option 1"}},
          {{"label": "B", "text": "Option 2"}},
          {{"label": "C", "text": "Option 3"}},
          {{"label": "D", "text": "Option 4"}}
        ],
        "answer_text": "Option 2"
      }}
    ]
    
    Do NOT include markdown formatting wrappers like ```json at the beginning, just return the raw JSON array.
    
    TEXT:
    \"\"\"
    {text}
    \"\"\"
    """

    try:
        # Use a fast, solid model suitable for JSON generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        raw_output = response.text.strip()
        
        # Clean markdown code blocks if the model ignored our formatting instruction
        if raw_output.startswith("```"):
            # strip ```json and ```
            raw_output = re.sub(r'^```[\w]*\s*', '', raw_output)
            raw_output = re.sub(r'\s*```$', '', raw_output)
            
        mcqs = json.loads(raw_output)
        
        # Validate structure to prevent 500 crashes down the line
        valid_mcqs = []
        for q in mcqs:
            if "question" in q and "options" in q and "answer_text" in q:
                if len(q["options"]) == 4:
                    valid_mcqs.append(q)
                    
        return valid_mcqs[:num_questions]
        
    except Exception as e:
        print(f"[ERROR] Failed to generate MCQs with Gemini: {e}")
        return []
