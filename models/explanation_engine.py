# models/explanation_engine.py
import json
import requests
from config import GROK_API_KEY, GROK_MODEL

def explain_answers(details):
    """
    Generate highly-personalized explanations for incorrect student answers using Grok API.
    
    details: list of dicts with question, selected, correct, is_correct
    Returns a list of explanation strings.
    """
    if not GROK_API_KEY:
        return [
            f"✅ Correct!" if d["is_correct"] else f"❌ Incorrect. The answer is '{d['correct']}'."
            for d in details
        ]

    explanations = []
    
    # We only want to ask Grok to explain the questions the student got WRONG
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
            
    prompt += 'You MUST respond with a JSON object containing an "explanations" key, which holds a flat array of strings, in the exact same order as the questions above. Format exactly like this: {"explanations": ["Explanation for Q1", "Explanation for Q2"]}'
    
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful teacher that only outputs valid JSON objects."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3
    }

    try:
        response = requests.post(
            "https://api.xai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        
        if isinstance(data, dict) and "explanations" in data:
            ai_explanations = data["explanations"]
        elif isinstance(data, list):
            ai_explanations = data
        else:
            ai_explanations = []
        
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
