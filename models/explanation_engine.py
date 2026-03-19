# models/explanation_engine.py
import json
import re
import google.generativeai as genai
from config import GEMINI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def explain_answers(details):
    """
    Generate highly-personalized explanations for incorrect student answers using Gemini API.
    
    details: list of dicts with question, selected, correct, is_correct
    Returns a list of explanation strings.
    """
    if not GEMINI_API_KEY:
        return [
            f"✅ Correct!" if d["is_correct"] else f"❌ Incorrect. The answer is '{d['correct']}'."
            for d in details
        ]

    explanations = []
    
    # We only want to ask Gemini to explain the questions the student got WRONG
    ai_questions = []
    for idx, d in enumerate(details):
        if not d["is_correct"] and d["selected"]:
            ai_questions.append((idx, d))
            
    if not ai_questions:
        for d in details:
            if d["is_correct"]:
                explanations.append(f"✅ Correct! '{d['selected']}' is exactly right.")
            else:
                explanations.append(f"⏭️ Not answered. The correct answer is '{d['correct']}'.")
        return explanations
        
    prompt = "You are a helpful teacher. Briefly explain WHY the student's selected wrong answer is incorrect, and WHY the correct answer is right. Keep it to exactly 1 or 2 clear sentences per question.\n\n"
    
    for _, d in ai_questions:
        prompt += f"Question: {d['question']}\nStudent Answer: {d['selected']}\nCorrect Answer: {d['correct']}\n\n"
            
    prompt += "Return your response as a pure JSON flat array of strings, in the exact same order as the questions above. Format exactly like this: [\"Explanation for Q1\", \"Explanation for Q2\"]\nDo not use a JSON wrapper, just return the raw array."
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        raw = response.text.strip()
        
        if raw.startswith("```"):
            raw = re.sub(r'^```[\w]*\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            
        ai_explanations = json.loads(raw)
        
        # Merge them back together in order
        ai_idx = 0
        for d in details:
            if d["is_correct"]:
                explanations.append(f"✅ Correct! '{d['selected']}' is exactly right.")
            elif not d["selected"]:
                explanations.append(f"⏭️ Not answered. The correct answer is '{d['correct']}'.")
            else:
                if ai_idx < len(ai_explanations):
                    explanations.append(f"❌ Incorrect. {ai_explanations[ai_idx]}")
                    ai_idx += 1
                else:
                    explanations.append(f"❌ Incorrect. The correct answer is '{d['correct']}'.")
                    
        return explanations
    except Exception as e:
        print(f"[ERROR] Failed to generate AI explanations: {e}")
        return [
            f"✅ Correct!" if d["is_correct"] else f"❌ Incorrect. The answer is '{d['correct']}'."
            for d in details
        ]
