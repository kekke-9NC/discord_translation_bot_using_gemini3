import asyncio
import builtins
import importlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import gemini_settings


class GeminiSettingsTests(unittest.TestCase):
    def test_default_model_and_environment_override(self):
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(gemini_settings)
            self.assertEqual(gemini_settings.GEMINI_MODEL, "gemma-4-26b-a4b-it")

        with patch.dict(os.environ, {"GEMINI_MODEL": "custom-model"}, clear=True):
            importlib.reload(gemini_settings)
            self.assertEqual(gemini_settings.GEMINI_MODEL, "custom-model")

        # Restore the module value for any subsequent tests in this process.
        importlib.reload(gemini_settings)

    def test_translation_and_reply_requests_share_the_configured_model(self):
        from discord.ext import commands
        from google import genai

        database = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        database.close()
        actual_open = builtins.open
        fake_async_client = SimpleNamespace(models=SimpleNamespace())

        def open_without_credentials(file, *args, **kwargs):
            if str(file).endswith(("discord_token.txt", "gemini_api_key.txt")):
                raise FileNotFoundError(str(file))
            return actual_open(file, *args, **kwargs)

        try:
            with patch.dict(os.environ, {"BOT_DATABASE": database.name, "GEMINI_MODEL": "custom-model"}), \
                 patch.object(commands.Bot, "run"), \
                 patch.object(genai, "Client", return_value=SimpleNamespace(aio=fake_async_client)), \
                 patch("builtins.open", side_effect=open_without_credentials):
                importlib.reload(gemini_settings)
                bot_module = importlib.import_module("bot")

            generated = AsyncMock(return_value=SimpleNamespace(text="generated text"))
            with patch.object(bot_module.async_client.models, "generate_content", generated, create=True):
                translated = asyncio.run(bot_module.translate_text("hello", "English", "Japanese"))

                interaction = Mock()
                interaction.response.defer = AsyncMock()
                interaction.followup.send = AsyncMock()
                view = bot_module.SuggestReplyView(123)
                asyncio.run(view.children[0].callback(interaction))

            self.assertEqual(translated, "generated text")
            self.assertEqual(generated.await_count, 2)
            self.assertEqual([call.kwargs["model"] for call in generated.await_args_list],
                             ["custom-model"] * 2)
            thinking_levels = [call.kwargs["config"].thinking_config.thinking_level.value.lower()
                               for call in generated.await_args_list]
            self.assertEqual(thinking_levels, ["minimal", "minimal"])
        finally:
            importlib.reload(gemini_settings)
            os.unlink(database.name)


if __name__ == "__main__":
    unittest.main()
