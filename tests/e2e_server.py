"""Local browser-test server. Temporary DB and stubbed AI; never deploy this module."""
import os
import tempfile

_data = tempfile.TemporaryDirectory(prefix='examly_e2e_')
for _key in ('VERCEL', 'RENDER'):
    os.environ.pop(_key, None)
os.environ.update({
    'DB_PATH': os.path.join(_data.name, 'test.db'),
    'DATABASE_URL': '', 'SECRET_KEY': 'browser-tests-only',
    'EMBEDDING_BACKEND': 'hashing', 'VECTOR_STORE': 'sqlite',
    'LLM_API_KEY': '', 'GROQ_API_KEY': '', 'GROK_API_KEY': '',
    'OPENAI_API_KEY': '', 'ANTHROPIC_API_KEY': '', 'GEMINI_API_KEY': '',
    'REDIS_URL': '', 'GENERATION_BACKEND': 'inline',
    'GENERATION_COOLDOWN_SECONDS': '0', 'GENERATION_MAX_PER_5MIN': '0',
})

import config
import core.services.demo_service as demo
import core.services.generation_status as status
import core.services.mcq_pipeline as pipeline
import models.explanation_engine as explanations
import models.mcq_generator as generator
from core.models.questions import attach_evidence
from core.repositories import document_repo

config.DATABASE_URL = ''
status.generation_available = lambda: True
demo.SAMPLE_TITLE = 'Cell Biology - Browser Test Material'
demo.SAMPLE_TEXT = 'Mitochondria produce ATP through cellular respiration. ' * 20


def questions(*args, **kwargs):
    document_id = args[0] if args and isinstance(args[0], int) else None
    source = (' '.join(chunk['content'] for chunk in document_repo.get_chunks(document_id))
              if document_id else args[0])
    result = [{
        'question': f'Question {i + 1}: Which organelle produces ATP?',
        'options': [{'label': label, 'text': text} for label, text in zip(
            'ABCD', ['Mitochondria', 'Nucleus', 'Ribosomes', 'Chloroplasts'], strict=True)],
        'answer_text': 'Mitochondria',
        'explanation': 'Mitochondria produce ATP through cellular respiration.',
        'source_hint': 'Mitochondria produce ATP',
    } for i in range(2)]
    return attach_evidence(result, source, document_id=document_id)


pipeline.generate_from_document = questions
pipeline.generate_from_text = questions
generator.generate_mcqs = questions
explanations.explain_answers = lambda details, **kwargs: ['Mitochondria produce ATP through cellular respiration.'] * len(details)

if __name__ == '__main__':
    import uvicorn

    from fastapi_app.main import app

    uvicorn.run(app, host='127.0.0.1', port=8765)
