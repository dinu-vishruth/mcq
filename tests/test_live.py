"""Live, credential-gated tests.

These talk to a real LLM provider and cost tokens, so they are deselected by
default (see pyproject.toml: ``-m 'not live'``). Run explicitly with::

    pytest -m live

They are also skipped automatically when no provider key is present, so a
plain ``pytest -m live`` on a machine without credentials exits cleanly.
"""
import os
import unittest

import pytest

pytestmark = pytest.mark.live


_HAS_KEY = any(os.getenv(name) for name in
               ("GROQ_API_KEY", "GROK_API_KEY", "XAI_API_KEY",
                "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
                "LLM_API_KEY"))


@unittest.skipUnless(_HAS_KEY, "No LLM provider credentials in environment")
class TestLiveGeneration(unittest.TestCase):
    def test_generate_mcqs_returns_valid_questions(self):
        from core.services.mcq_pipeline import generate_mcqs_rag

        document = (
            "Photosynthesis converts sunlight, water and carbon dioxide into glucose "
            "and oxygen inside chloroplasts. Chlorophyll captures the light energy. "
            "The process has two stages: the light-dependent reactions and the "
            "Calvin cycle, which fixes carbon dioxide into sugar."
        )
        mcqs = generate_mcqs_rag(document, num_questions=3, difficulty="easy")
        self.assertGreaterEqual(len(mcqs), 1)
        for mcq in mcqs:
            self.assertTrue(mcq["question"].strip())
            self.assertEqual(len(mcq["options"]), 4)
            self.assertIn(mcq["answer_text"], [o["text"] for o in mcq["options"]])

    def test_model_resolution_builds_a_provider(self):
        from core.llm import get_llm

        llm = get_llm()
        self.assertTrue(llm.provider_id)
        self.assertTrue(llm.model)
