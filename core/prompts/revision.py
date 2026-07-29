"""
Revision prompts. The RevisionAgent turns a student's weak topics + retrieved
source context into concise revision notes, grounded in the original material.
"""

SYSTEM = "You are a helpful study coach that only outputs valid JSON objects."


def revision_notes_prompt(topics: list[str], context: str) -> str:
    topic_list = ", ".join(topics) if topics else "the key concepts"
    return f"""Using ONLY the CONTEXT below, write concise revision notes focused on: {topic_list}.
Do not invent facts beyond the CONTEXT.

Respond with a JSON object exactly like:
{{"notes": [{{"topic": "Topic name", "summary": "2-3 sentence revision summary", "key_points": ["point 1", "point 2"]}}]}}

CONTEXT:
\"\"\"
{context}
\"\"\"
"""
