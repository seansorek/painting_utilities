"""Tests for _post_scheduled_challenges and _send_daily_challenge (Issue #11).

Bug A: A malformed/missing post_at must not abort the tick for other entries.
Bug B: A failed post must keep the entry in the schedule; only a successful
       post should remove the entry.
"""

import json
import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stub for the discord module so bot.py can be imported without a live
# Discord connection or py-cord installed in the test environment.
# ---------------------------------------------------------------------------

def _make_discord_stub():
    discord_mod = types.ModuleType("discord")

    class _Intents:
        @staticmethod
        def default():
            return _Intents()

    class _Bot:
        def __init__(self, **kwargs):
            self._guilds = {}

        def event(self, fn):
            return fn

        def slash_command(self, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        def get_guild(self, guild_id):
            return self._guilds.get(guild_id)

        def run(self, token):
            pass

    class _Color:
        @staticmethod
        def blurple():
            return None
        @staticmethod
        def from_rgb(*args):
            return None

    class _Embed:
        def __init__(self, **kwargs):
            pass
        def add_field(self, **kwargs):
            pass
        def set_image(self, **kwargs):
            pass
        def set_thumbnail(self, **kwargs):
            pass
        def set_footer(self, **kwargs):
            pass

    class _File:
        def __init__(self, *args, **kwargs):
            pass

    class _Attachment:
        pass

    class _ApplicationContext:
        pass

    def _Option(type_, **kwargs):
        return None

    discord_mod.Bot = _Bot
    discord_mod.Intents = _Intents
    discord_mod.Color = _Color
    discord_mod.Embed = _Embed
    discord_mod.File = _File
    discord_mod.Attachment = _Attachment
    discord_mod.ApplicationContext = _ApplicationContext
    discord_mod.Option = _Option
    return discord_mod


def _make_analyzer_stub():
    """Return a minimal stub for the analyzer module."""
    mod = types.ModuleType("analyzer")
    for name in [
        "load_image_from_bytes", "extract_dominant_colors", "compute_stats",
        "render_palette_chart", "render_hue_saturation_chart", "render_chart_to_bytesio",
        "nearest_color_name", "apply_gradient_map", "GRADIENT_PRESETS", "parse_hex_color",
        "parse_multi_hex_gradient", "reverse_gradient", "render_gradient_preview",
        "rgb_to_cmyk", "classify_palette_type", "palette_to_gradient_stops",
        "adjust_image", "simulate_colorblindness", "render_colorblind_comparison",
        "recolor_image", "suggest_harmony_colors", "render_harmony_chart",
        "render_color_info_swatch", "render_compare_chart", "export_ase", "export_swatches",
        "export_gpl", "export_aco", "export_css", "export_tailwind",
        "export_gradient_ggr", "export_gradient_json",
    ]:
        setattr(mod, name, MagicMock())
    mod.GRADIENT_PRESETS = {}
    return mod


# Install stubs before importing bot
sys.modules.setdefault("discord", _make_discord_stub())
sys.modules.setdefault("analyzer", _make_analyzer_stub())
sys.modules.setdefault("dotenv", types.ModuleType("dotenv"))
sys.modules["dotenv"].load_dotenv = lambda: None

# Now import the module under test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bot as bot_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _future_iso(hours: int = 1) -> str:
    """Return an ISO-8601 string that is `hours` hours in the future (UTC)."""
    return (datetime.now(tz=timezone.utc) + timedelta(hours=hours)).isoformat()


def _past_iso(hours: int = 1) -> str:
    """Return an ISO-8601 string that is `hours` hours in the past (UTC)."""
    return (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPostScheduledChallengesBugA(unittest.IsolatedAsyncioTestCase):
    """Bug A: malformed post_at must not abort the tick for other entries."""

    async def asyncSetUp(self):
        # Patch schedule I/O so no real files are touched
        self._load_patch = patch.object(bot_module, "_load_schedule")
        self._save_patch = patch.object(bot_module, "_save_schedule")
        self.mock_load = self._load_patch.start()
        self.mock_save = self._save_patch.start()

        # Patch _send_daily_challenge to always succeed
        self._send_patch = patch.object(bot_module, "_send_daily_challenge", new=AsyncMock(return_value=True))
        self.mock_send = self._send_patch.start()

    async def asyncTearDown(self):
        self._load_patch.stop()
        self._save_patch.stop()
        self._send_patch.stop()

    async def test_malformed_post_at_is_skipped(self):
        """An entry with a bad post_at is dropped without raising, tick continues."""
        bad_entry = {"guild_id": "1", "channel_id": "10", "content": "bad", "post_at": "NOT-A-DATE"}
        good_entry = {"guild_id": "2", "channel_id": "20", "content": "good", "post_at": _past_iso()}

        self.mock_load.return_value = [bad_entry, good_entry]

        # Must not raise
        await bot_module._post_scheduled_challenges()

        # The good entry was successfully delivered, so remaining should be empty
        saved = self.mock_save.call_args[0][0]
        self.assertEqual(saved, [], "Successfully-delivered good entry should be pruned")
        # The bad entry was dropped silently (not kept in remaining)
        for entry in saved:
            self.assertNotEqual(entry["content"], "bad")

    async def test_missing_post_at_key_is_skipped(self):
        """An entry missing the post_at key entirely is dropped without raising."""
        bad_entry = {"guild_id": "1", "channel_id": "10", "content": "no-key"}
        self.mock_load.return_value = [bad_entry]

        await bot_module._post_scheduled_challenges()

        saved = self.mock_save.call_args[0][0]
        self.assertEqual(saved, [])

    async def test_future_entry_is_preserved_regardless_of_bad_peers(self):
        """A future-dated entry survives even when a peer has a bad post_at."""
        bad_entry = {"guild_id": "1", "channel_id": "10", "content": "bad", "post_at": "INVALID"}
        future_entry = {"guild_id": "2", "channel_id": "20", "content": "future", "post_at": _future_iso(2)}

        self.mock_load.return_value = [bad_entry, future_entry]
        await bot_module._post_scheduled_challenges()

        saved = self.mock_save.call_args[0][0]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["content"], "future")


class TestPostScheduledChallengesBugB(unittest.IsolatedAsyncioTestCase):
    """Bug B: failed posts must stay in the schedule; successful posts must be pruned."""

    async def asyncSetUp(self):
        self._load_patch = patch.object(bot_module, "_load_schedule")
        self._save_patch = patch.object(bot_module, "_save_schedule")
        self.mock_load = self._load_patch.start()
        self.mock_save = self._save_patch.start()

    async def asyncTearDown(self):
        self._load_patch.stop()
        self._save_patch.stop()

    async def test_failed_post_stays_in_schedule(self):
        """When _send_daily_challenge returns False, the entry is kept for retry."""
        entry = {"guild_id": "1", "channel_id": "10", "content": "retry-me", "post_at": _past_iso()}
        self.mock_load.return_value = [entry]

        with patch.object(bot_module, "_send_daily_challenge", new=AsyncMock(return_value=False)):
            await bot_module._post_scheduled_challenges()

        saved = self.mock_save.call_args[0][0]
        self.assertEqual(len(saved), 1, "Failed entry must remain for retry")
        self.assertEqual(saved[0]["content"], "retry-me")

    async def test_successful_post_is_removed_from_schedule(self):
        """When _send_daily_challenge returns True, the entry is pruned."""
        entry = {"guild_id": "1", "channel_id": "10", "content": "done", "post_at": _past_iso()}
        self.mock_load.return_value = [entry]

        with patch.object(bot_module, "_send_daily_challenge", new=AsyncMock(return_value=True)):
            await bot_module._post_scheduled_challenges()

        saved = self.mock_save.call_args[0][0]
        self.assertEqual(saved, [], "Successfully-posted entry must be pruned")

    async def test_future_entry_is_not_sent_and_stays_in_schedule(self):
        """An entry whose post_at is in the future is left untouched."""
        entry = {"guild_id": "1", "channel_id": "10", "content": "later", "post_at": _future_iso(5)}
        self.mock_load.return_value = [entry]

        with patch.object(bot_module, "_send_daily_challenge", new=AsyncMock(return_value=True)) as mock_send:
            await bot_module._post_scheduled_challenges()
            mock_send.assert_not_called()

        saved = self.mock_save.call_args[0][0]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["content"], "later")

    async def test_mixed_success_and_failure(self):
        """Only the successful entry is pruned; the failed one remains."""
        success_entry = {"guild_id": "1", "channel_id": "10", "content": "ok", "post_at": _past_iso(1)}
        fail_entry = {"guild_id": "2", "channel_id": "20", "content": "fail", "post_at": _past_iso(1)}
        self.mock_load.return_value = [success_entry, fail_entry]

        async def _selective_send(challenge):
            return challenge["content"] == "ok"

        with patch.object(bot_module, "_send_daily_challenge", new=_selective_send):
            await bot_module._post_scheduled_challenges()

        saved = self.mock_save.call_args[0][0]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["content"], "fail")


class TestPostScheduledChallengesExpiry(unittest.IsolatedAsyncioTestCase):
    """Entries far past their post_at (beyond CHALLENGE_EXPIRY_HOURS) are dropped."""

    async def asyncSetUp(self):
        self._load_patch = patch.object(bot_module, "_load_schedule")
        self._save_patch = patch.object(bot_module, "_save_schedule")
        self.mock_load = self._load_patch.start()
        self.mock_save = self._save_patch.start()

    async def asyncTearDown(self):
        self._load_patch.stop()
        self._save_patch.stop()

    async def test_expired_entry_is_dropped_even_on_failure(self):
        """An entry more than CHALLENGE_EXPIRY_HOURS overdue is dropped regardless."""
        overdue_hours = bot_module._CHALLENGE_EXPIRY_HOURS + 1
        expired_entry = {
            "guild_id": "1", "channel_id": "10", "content": "old",
            "post_at": _past_iso(overdue_hours),
        }
        self.mock_load.return_value = [expired_entry]

        with patch.object(bot_module, "_send_daily_challenge", new=AsyncMock(return_value=False)):
            await bot_module._post_scheduled_challenges()

        saved = self.mock_save.call_args[0][0]
        self.assertEqual(saved, [], "Expired entry must be dropped even if delivery failed")


class TestSendDailyChallenge(unittest.IsolatedAsyncioTestCase):
    """Unit tests for _send_daily_challenge return values."""

    def _make_bot_with_guild(self, guild_id, channel_id, channel=None):
        mock_guild = MagicMock()
        mock_guild.get_channel.return_value = channel
        bot_module.bot._guilds = {guild_id: mock_guild}
        return mock_guild

    async def asyncTearDown(self):
        bot_module.bot._guilds = {}

    async def test_returns_false_when_guild_not_found(self):
        bot_module.bot._guilds = {}
        result = await bot_module._send_daily_challenge(
            {"guild_id": "999", "channel_id": "1", "content": "hi", "post_at": _past_iso()}
        )
        self.assertFalse(result)

    async def test_returns_false_when_channel_not_found(self):
        self._make_bot_with_guild(42, 99, channel=None)
        result = await bot_module._send_daily_challenge(
            {"guild_id": "42", "channel_id": "99", "content": "hi", "post_at": _past_iso()}
        )
        self.assertFalse(result)

    async def test_returns_true_on_successful_send(self):
        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock()
        self._make_bot_with_guild(42, 99, channel=mock_channel)

        result = await bot_module._send_daily_challenge(
            {"guild_id": "42", "channel_id": "99", "content": "hello", "post_at": _past_iso()}
        )
        self.assertTrue(result)
        mock_channel.send.assert_awaited_once_with("hello")

    async def test_returns_false_on_discord_api_error(self):
        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock(side_effect=Exception("Discord API error"))
        self._make_bot_with_guild(42, 99, channel=mock_channel)

        result = await bot_module._send_daily_challenge(
            {"guild_id": "42", "channel_id": "99", "content": "hi", "post_at": _past_iso()}
        )
        self.assertFalse(result)

    async def test_returns_false_when_guild_id_missing(self):
        result = await bot_module._send_daily_challenge(
            {"channel_id": "99", "content": "hi", "post_at": _past_iso()}
        )
        self.assertFalse(result)

    async def test_returns_false_when_channel_id_missing(self):
        result = await bot_module._send_daily_challenge(
            {"guild_id": "1", "content": "hi", "post_at": _past_iso()}
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
