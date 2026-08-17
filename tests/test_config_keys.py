"""
Regression net for LLM credential resolution.

These lock the two bugs that produced "API Key is missing" while a valid key was
sitting in .env:

  1. A variable that EXISTS but is BLANK shadowed its fallback, because
     os.getenv(name, fallback) only falls back when the name is absent. Adding
     LLM_API_KEY with an empty value (easy to do in a hosting dashboard) blanked
     out a working GROQ_API_KEY.
  2. Only the GROK_* spelling was read, though a `gsk_...` key is a *Groq* key
     and GROQ_* is the natural spelling for it.

Also covers provider auto-detection, which previously resolved to xAI with an
empty key whenever only a provider-specific key was configured.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.llm import registry


class TestEnvStr(unittest.TestCase):
    """config.env_str: blank means unset, and later names are fallbacks."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("T_ONE", "T_TWO", "LLM_API_KEY", "GROK_API_KEY", "GROQ_API_KEY")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_blank_does_not_shadow_later_name(self):
        os.environ["T_ONE"] = ""          # present but empty
        os.environ["T_TWO"] = "real"
        self.assertEqual(config.env_str("T_ONE", "T_TWO"), "real")

    def test_blank_does_not_shadow_default(self):
        os.environ["T_ONE"] = "   "       # whitespace only
        self.assertEqual(config.env_str("T_ONE", default="fallback"), "fallback")

    def test_first_non_blank_wins(self):
        os.environ["T_ONE"] = "first"
        os.environ["T_TWO"] = "second"
        self.assertEqual(config.env_str("T_ONE", "T_TWO"), "first")

    def test_strips_whitespace_and_quotes(self):
        # A copy-pasted `KEY="abc"` line, or a value with a trailing space.
        os.environ["T_ONE"] = '  "gsk_abc"  '
        self.assertEqual(config.env_str("T_ONE"), "gsk_abc")
        os.environ["T_ONE"] = "'gsk_xyz'"
        self.assertEqual(config.env_str("T_ONE"), "gsk_xyz")

    def test_absent_returns_default(self):
        self.assertEqual(config.env_str("T_NOPE", default="d"), "d")
        self.assertEqual(config.env_str("T_NOPE"), "")

    def test_groq_spelling_is_accepted(self):
        os.environ["GROQ_API_KEY"] = "gsk_qspelling"
        self.assertEqual(
            config.env_str("GROK_API_KEY", "GROQ_API_KEY", "XAI_API_KEY"),
            "gsk_qspelling",
        )

    def test_blank_llm_key_does_not_shadow_groq_key(self):
        """The exact reported failure: blank LLM_API_KEY + good Groq key."""
        os.environ["LLM_API_KEY"] = ""
        os.environ["GROQ_API_KEY"] = "gsk_realkey"
        groq = config.env_str("GROK_API_KEY", "GROQ_API_KEY", "XAI_API_KEY")
        self.assertEqual(config.env_str("LLM_API_KEY", default=groq), "gsk_realkey")


class TestKeyPresence(unittest.TestCase):
    def setUp(self):
        self._saved = {n: getattr(config, n) for n in
                       ("LLM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")}

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(config, n, v)

    def _clear(self):
        for n in self._saved:
            setattr(config, n, "")

    def test_no_keys_means_absent(self):
        self._clear()
        self.assertFalse(config.llm_key_present())

    def test_any_single_key_counts(self):
        for name in self._saved:
            self._clear()
            setattr(config, name, "some-key")
            self.assertTrue(config.llm_key_present(), f"{name} should satisfy presence")

    def test_missing_key_message_is_actionable(self):
        msg = config.missing_key_message()
        # Names the variable to set and where it looked, and stays ASCII so the
        # Windows console (cp1252) doesn't mangle it.
        self.assertIn("GROQ_API_KEY", msg)
        self.assertTrue(any(t in msg for t in (".env", "environment variables")))
        self.assertTrue(all(ord(c) < 128 for c in msg), "message must be ASCII-safe")


class TestProviderAutoDetection(unittest.TestCase):
    """LLM_PROVIDER=auto must never pick a provider whose key is empty."""

    def setUp(self):
        self._saved = {n: getattr(config, n) for n in
                       ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")}
        for n in self._saved:
            setattr(config, n, "")

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(config, n, v)

    def test_groq_prefix(self):
        self.assertEqual(registry._resolve_auto("gsk_x", "llama-3.3-70b-versatile")[0], "groq")

    def test_groq_key_overrides_stale_grok_model(self):
        # A leftover LLM_MODEL=grok-... must not be sent to Groq, which would 404.
        provider, model = registry._resolve_auto("gsk_x", "grok-2-1212")
        self.assertEqual(provider, "groq")
        self.assertEqual(model, "groq/compound-mini")

    def test_xai_and_openai_and_anthropic_prefixes(self):
        self.assertEqual(registry._resolve_auto("xai-x", "grok-2-1212")[0], "xai")
        self.assertEqual(registry._resolve_auto("sk-ant-x", "")[0], "anthropic")
        self.assertEqual(registry._resolve_auto("sk-x", "")[0], "openai")

    def test_falls_back_to_provider_specific_key(self):
        """No unified key: use the provider whose own key is set, not xAI."""
        config.ANTHROPIC_API_KEY = "sk-ant-x"
        provider, model = registry._resolve_auto("", "groq/compound-mini")
        self.assertEqual(provider, "anthropic")
        self.assertIn("claude", model)   # stale groq model name not reused

        config.ANTHROPIC_API_KEY = ""
        config.GEMINI_API_KEY = "gk-x"
        self.assertEqual(registry._resolve_auto("", "")[0], "gemini")

    def test_model_kept_when_plausible_for_provider(self):
        _, model = registry._resolve_auto("gsk_x", "qwen/qwen3.6-27b")
        self.assertEqual(model, "qwen/qwen3.6-27b")


class TestGenerationGuard(unittest.TestCase):
    def test_generate_mcqs_raises_user_safe_message(self):
        """With no key, the learner-facing error must not leak server config.

        The message reaches the browser via the upload template, so it must
        contain neither variable names nor filesystem paths.
        """
        import models.mcq_generator as mcq

        saved = {n: getattr(config, n) for n in
                 ("LLM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")}
        for n in saved:
            setattr(config, n, "")
        try:
            with self.assertRaises(mcq.MCQGenerationError) as ctx:
                mcq.generate_mcqs("Some study text.", num_questions=2)
            shown = str(ctx.exception)
            self.assertEqual(shown, config.USER_FACING_AI_UNAVAILABLE)
            for leak in ("GROQ_API_KEY", "GROK_API_KEY", "LLM_API_KEY", ".env", "C:\\", "/var"):
                self.assertNotIn(leak, shown, f"user-facing text leaked {leak!r}")
        finally:
            for n, v in saved.items():
                setattr(config, n, v)


if __name__ == "__main__":
    unittest.main(verbosity=2)
