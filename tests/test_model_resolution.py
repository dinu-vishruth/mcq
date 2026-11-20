"""Provider/model resolution.

Every provider has its own default model, a model belonging to one provider is
never sent to another, and an unrecoverable mismatch fails fast (LLMError)
rather than at request time.
"""
import unittest
from unittest.mock import Mock, patch

import config
from core.llm import registry
from core.llm.base import LLMError


class TestProviderDefaults(unittest.TestCase):
    def test_each_provider_has_its_own_default(self):
        expected = {"groq", "xai", "openai", "gemini", "anthropic"}
        self.assertEqual(set(config.PROVIDER_DEFAULT_MODELS), expected)
        for provider in expected:
            with self.subTest(provider=provider):
                self.assertTrue(config.default_model_for(provider))

    def test_unknown_provider_has_no_default(self):
        self.assertEqual(config.default_model_for("mystery"), "")

    def test_unset_model_uses_provider_default(self):
        """LLM_MODEL empty must not fall back to GROK_MODEL ("grok-2-1212") for
        a Groq key -- that was the original cross-provider bug."""
        cases = [
            ("groq", "gsk_test", "openai/gpt-oss-20b"),
            ("xai", "xai-test", "grok-2-1212"),
            ("openai", "sk-test", "gpt-4o-mini"),
            ("gemini", "gm-test", "gemini-1.5-flash"),
            ("anthropic", "sk-ant-test", "claude-sonnet-5"),
        ]
        for provider, key, expected_model in cases:
            with self.subTest(provider=provider), patch.multiple(
                config, LLM_PROVIDER=provider, LLM_API_KEY=key,
                LLM_MODEL="", OPENAI_API_KEY=key, GEMINI_API_KEY=key,
                ANTHROPIC_API_KEY=key,
            ):
                llm = registry.build_llm()
                self.assertEqual(llm.provider_id, provider)
                self.assertEqual(llm.model, expected_model)

    def test_foreign_model_is_replaced_with_provider_default(self):
        for provider in ("groq", "xai", "openai", "gemini", "anthropic"):
            for foreign in ("grok-2-1212", "openai/gpt-oss-20b", "gemini-1.5-flash"):
                expected = config.default_model_for(provider)
                if expected == foreign:
                    continue
                with self.subTest(provider=provider, model=foreign), patch.multiple(
                    config, LLM_PROVIDER=provider, LLM_API_KEY="sk-any",
                    LLM_MODEL=foreign, OPENAI_API_KEY="sk-any",
                    GEMINI_API_KEY="gm-any", ANTHROPIC_API_KEY="sk-ant-any",
                ):
                    self.assertEqual(registry.build_llm().model, expected)

    def test_matching_model_is_preserved(self):
        with patch.multiple(
            config, LLM_PROVIDER="openai", LLM_API_KEY="sk-test",
            LLM_MODEL="gpt-4.1-mini",
        ):
            llm = registry.build_llm()
            self.assertEqual(llm.model, "gpt-4.1-mini")

    def test_auto_groq_key_ignores_stale_xai_model(self):
        provider, model = registry._resolve_auto("gsk_x", "grok-2-1212")
        self.assertEqual((provider, model), ("groq", "openai/gpt-oss-20b"))

    def test_retired_compound_models_use_default_in_json_request(self):
        for provider in ("auto", "groq"):
            for retired in ("groq/compound-mini", "groq/compound"):
                with self.subTest(provider=provider, model=retired), patch.multiple(
                    config, LLM_PROVIDER=provider, LLM_API_KEY="gsk_test",
                    LLM_MODEL=retired,
                ), patch("core.llm.openai_compatible.requests.post") as post:
                    post.return_value = Mock(status_code=200)
                    post.return_value.json.return_value = {
                        "choices": [{"message": {"content": '{"questions": []}'}}],
                    }
                    llm = registry.build_llm()
                    result = llm.complete_json([
                        {"role": "user", "content": "Return questions as JSON."},
                    ])
                    self.assertEqual(result, {"questions": []})
                    self.assertEqual(post.call_args.args[0],
                                     "https://api.groq.com/openai/v1/chat/completions")
                    self.assertEqual(post.call_args.kwargs["json"]["model"],
                                     "openai/gpt-oss-20b")

    def test_active_groq_models_remain_selectable(self):
        for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
            with self.subTest(model=model), patch.multiple(
                config, LLM_PROVIDER="groq", LLM_API_KEY="gsk_test", LLM_MODEL=model,
            ):
                self.assertEqual(registry.build_llm().model, model)


class TestValidation(unittest.TestCase):
    def test_valid_pair_passes(self):
        registry.validate_provider_model("groq", "openai/gpt-oss-20b")
        registry.validate_provider_model("xai", "grok-2-1212")

    def test_foreign_model_raises_with_actionable_message(self):
        with self.assertRaises(LLMError) as ctx:
            registry.validate_provider_model("groq", "grok-2-1212")
        message = str(ctx.exception)
        self.assertIn("groq", message)
        self.assertIn("LLM_MODEL", message)
        self.assertIn("openai/gpt-oss-20b", message)

    def test_unknown_provider_raises(self):
        with self.assertRaises(LLMError):
            registry.validate_provider_model("mystery", "whatever")

    def test_build_llm_rejects_unknown_provider(self):
        with patch.multiple(
            config, LLM_PROVIDER="mystery", LLM_API_KEY="sk-test", LLM_MODEL=""
        ):
            with self.assertRaises(LLMError):
                registry.build_llm()


if __name__ == "__main__":
    unittest.main(verbosity=2)
