"""Tests for _post_scheduled_challenges and _send_daily_challenge (Issue #11).

Bug A: A malformed/missing post_at must not abort the tick for other entries.
Bug B: A failed post must keep the entry in the schedule; only a successful
       post should remove the entry.
"""

import asyncio
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

        def check(self, fn):
            return fn

        def get_channel(self, channel_id):
            return None

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

    class _ForumChannel:
        pass

    class _Member:
        pass

    class _Role:
        pass

    class _Thread:
        pass

    class _Message:
        pass

    def _Option(type_, **kwargs):
        return None

    def _default_permissions(**kwargs):
        def decorator(fn):
            return fn
        return decorator

    discord_mod.Bot = _Bot
    discord_mod.Intents = _Intents
    discord_mod.Color = _Color
    discord_mod.Embed = _Embed
    discord_mod.File = _File
    discord_mod.Attachment = _Attachment
    discord_mod.ApplicationContext = _ApplicationContext
    discord_mod.ForumChannel = _ForumChannel
    discord_mod.Member = _Member
    discord_mod.Role = _Role
    discord_mod.Thread = _Thread
    discord_mod.Message = _Message
    discord_mod.Option = _Option
    discord_mod.default_permissions = _default_permissions
    return discord_mod


def _make_ext_stub():
    """Stub for discord.ext.commands and discord.ext.tasks."""
    ext_mod = types.ModuleType("discord.ext")

    commands_mod = types.ModuleType("discord.ext.commands")

    class _CheckFailure(Exception):
        pass

    commands_mod.CheckFailure = _CheckFailure

    tasks_mod = types.ModuleType("discord.ext.tasks")

    class _LoopDecorator:
        """Mimics @tasks.loop -- stores the coro and exposes start/is_running."""
        def __init__(self, **kwargs):
            pass

        def __call__(self, fn):
            async def wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)

            wrapper.start = lambda: None
            wrapper.is_running = lambda: False

            def _error(error_fn):
                # Mimics @loop.error -- just stores the handler, no invocation
                # wiring needed for the unit tests.
                wrapper._error_handler = error_fn
                return error_fn

            wrapper.error = _error
            return wrapper

    tasks_mod.loop = _LoopDecorator

    ext_mod.commands = commands_mod
    ext_mod.tasks = tasks_mod
    return ext_mod, commands_mod, tasks_mod


def _make_analyzer_stub():
    """Return a minimal stub for the analyzer module."""
    mod = types.ModuleType("analyzer")
    for name in [
        "MAX_IMAGE_PIXELS",
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
    mod.MAX_IMAGE_PIXELS = 100_000_000
    return mod


# Install stubs before importing bot
_discord_stub = _make_discord_stub()
_ext_stub, _commands_stub, _tasks_stub = _make_ext_stub()
sys.modules.setdefault("discord", _discord_stub)
sys.modules.setdefault("discord.ext", _ext_stub)
sys.modules.setdefault("discord.ext.commands", _commands_stub)
sys.modules.setdefault("discord.ext.tasks", _tasks_stub)
sys.modules.setdefault("analyzer", _make_analyzer_stub())
sys.modules.setdefault("dotenv", types.ModuleType("dotenv"))
sys.modules["dotenv"].load_dotenv = lambda: None
sys.modules.setdefault("pytz", types.ModuleType("pytz"))
sys.modules["pytz"].timezone = lambda tz: timezone.utc  # simplify: treat ET as UTC for tests

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

        # _save_schedule is always called; verify the entry is preserved
        self.assertTrue(self.mock_save.called, "_save_schedule should be called")
        saved = self.mock_save.call_args[0][0]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["content"], "later")

    async def test_future_entry_with_id_is_not_sent_and_stays_in_schedule(self):
        """Regression test for #71: a future-dated entry that carries an `id`
        (as every entry created by /daily_challenge does) must survive a tick,
        not be silently dropped."""
        entry = {
            "id": "fut-1",
            "guild_id": "1",
            "channel_id": "10",
            "content": "later",
            "post_at": _future_iso(5),
        }
        self.mock_load.return_value = [entry]

        with patch.object(bot_module, "_send_daily_challenge", new=AsyncMock(return_value=True)) as mock_send:
            await bot_module._post_scheduled_challenges()
            mock_send.assert_not_called()

        self.assertTrue(self.mock_save.called, "_save_schedule should be called")
        saved = self.mock_save.call_args[0][0]
        self.assertEqual(len(saved), 1, "Future-dated entry with an id must be preserved")
        self.assertEqual(saved[0]["content"], "later")
        self.assertEqual(saved[0]["id"], "fut-1")

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

    async def test_returns_false_when_guild_id_missing(self):
        result = await bot_module._send_daily_challenge(
            {"channel_id": "99", "content": "hi", "post_at": _past_iso()}
        )
        self.assertFalse(result)

    async def test_returns_false_when_no_channel_configured(self):
        with patch.object(bot_module, "_get_guild_channel", return_value=None):
            result = await bot_module._send_daily_challenge(
                {"guild_id": "42", "content": "hi", "post_at": _past_iso()}
            )
        self.assertFalse(result)

    async def test_returns_false_when_channel_not_found(self):
        with patch.object(bot_module, "_get_guild_channel", return_value="99"), \
             patch.object(bot_module.bot, "get_channel", return_value=None):
            result = await bot_module._send_daily_challenge(
                {"guild_id": "42", "content": "hi", "post_at": _past_iso()}
            )
        self.assertFalse(result)

    async def test_returns_true_on_successful_send(self):
        mock_channel = AsyncMock()
        mock_channel.create_thread = AsyncMock()

        with patch.object(bot_module, "_get_guild_channel", return_value="99"), \
             patch.object(bot_module.bot, "get_channel", return_value=mock_channel):
            result = await bot_module._send_daily_challenge(
                {"guild_id": "42", "day": "Day 1", "content": "hello", "post_at": _past_iso()}
            )
        self.assertTrue(result)

    async def test_returns_false_on_discord_api_error(self):
        mock_channel = AsyncMock(spec=_discord_stub.ForumChannel)
        mock_channel.create_thread = AsyncMock(side_effect=Exception("Discord API error"))

        with patch.object(bot_module, "_get_guild_channel", return_value="99"), \
             patch.object(bot_module.bot, "get_channel", return_value=mock_channel):
            result = await bot_module._send_daily_challenge(
                {"guild_id": "42", "day": "Day 1", "content": "hi", "post_at": _past_iso()}
            )
        self.assertFalse(result)


class TestPostScheduledChallengesTOCTOU(unittest.IsolatedAsyncioTestCase):
    """Issue #46: concurrent additions during Phase 2 must not be lost.

    The race is: Phase 1 reads the schedule, Phase 2 releases the lock for
    network sends (during which /daily_challenge may append a new entry), and
    Phase 3 must preserve that new entry rather than overwriting with the
    Phase 1 snapshot.
    """

    async def asyncSetUp(self):
        self._load_patch = patch.object(bot_module, "_load_schedule")
        self._save_patch = patch.object(bot_module, "_save_schedule")
        self.mock_load = self._load_patch.start()
        self.mock_save = self._save_patch.start()

    async def asyncTearDown(self):
        self._load_patch.stop()
        self._save_patch.stop()

    async def test_concurrent_addition_preserved(self):
        """An entry added during Phase 2 (network sends) is kept in Phase 3."""
        existing_entry = {
            "id": "existing-1",
            "guild_id": "1",
            "channel_id": "10",
            "content": "due now",
            "post_at": _past_iso(1),
        }
        concurrent_entry = {
            "id": "concurrent-new",
            "guild_id": "2",
            "channel_id": "20",
            "content": "added while sending",
            "post_at": _future_iso(2),
        }

        # Phase 1 sees only the existing entry.
        # Phase 3 re-reads and sees both (simulating concurrent addition).
        self.mock_load.side_effect = [
            [existing_entry],            # Phase 1 read
            [existing_entry, concurrent_entry],  # Phase 3 re-read
        ]

        with patch.object(bot_module, "_send_daily_challenge", new=AsyncMock(return_value=True)):
            await bot_module._post_scheduled_challenges()

        saved = self.mock_save.call_args[0][0]
        # The existing entry was sent successfully -> removed.
        # The concurrent entry was NOT in Phase 1 -> preserved.
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["id"], "concurrent-new")
        self.assertEqual(saved[0]["content"], "added while sending")

    async def test_concurrent_addition_preserved_alongside_failed(self):
        """Both a failed retry entry and a concurrent addition survive Phase 3."""
        due_entry = {
            "id": "due-1",
            "guild_id": "1",
            "channel_id": "10",
            "content": "will fail",
            "post_at": _past_iso(1),
        }
        concurrent_entry = {
            "id": "concurrent-2",
            "guild_id": "3",
            "channel_id": "30",
            "content": "new",
            "post_at": _future_iso(3),
        }

        self.mock_load.side_effect = [
            [due_entry],
            [due_entry, concurrent_entry],
        ]

        with patch.object(bot_module, "_send_daily_challenge", new=AsyncMock(return_value=False)):
            await bot_module._post_scheduled_challenges()

        saved = self.mock_save.call_args[0][0]
        saved_ids = {c["id"] for c in saved}
        # Both the failed entry (retry) and the concurrent entry are kept.
        self.assertIn("due-1", saved_ids)
        self.assertIn("concurrent-2", saved_ids)


class TestPostScheduledChallengesIssue78(unittest.IsolatedAsyncioTestCase):
    """Issue #78: a malformed post_at on an ID-bearing entry must actually be
    dropped in Phase 3, not re-written to disk forever.

    Every entry created by /daily_challenge carries an `id`. Phase 1 logs and
    skips malformed entries without recording their id anywhere, so Phase 3's
    "not seen in Phase 1 -> concurrently added, keep it" branch used to fire
    for them, making the entry permanently un-droppable.
    """

    async def asyncSetUp(self):
        self._load_patch = patch.object(bot_module, "_load_schedule")
        self._save_patch = patch.object(bot_module, "_save_schedule")
        self.mock_load = self._load_patch.start()
        self.mock_save = self._save_patch.start()

    async def asyncTearDown(self):
        self._load_patch.stop()
        self._save_patch.stop()

    async def test_malformed_post_at_with_id_is_actually_dropped(self):
        """An ID-bearing entry with an unparseable post_at must be gone from
        the saved schedule after a single tick, not re-persisted."""
        bad_entry = {
            "id": "bad-post-at-1",
            "guild_id": "1",
            "channel_id": "10",
            "content": "malformed",
            "post_at": "not-a-date",
        }
        # Phase 1 and Phase 3 both re-read from disk; nothing else is added
        # concurrently, so the same entry is returned both times.
        self.mock_load.return_value = [bad_entry]

        await bot_module._post_scheduled_challenges()

        saved = self.mock_save.call_args[0][0]
        self.assertEqual(saved, [], "Malformed ID-bearing entry must be dropped, not kept forever")

    async def test_malformed_post_at_with_id_dropped_alongside_other_entries(self):
        """The malformed entry is dropped while unrelated entries are handled
        normally (future entry kept, due entry sent and pruned)."""
        bad_entry = {
            "id": "bad-post-at-2",
            "guild_id": "1",
            "channel_id": "10",
            "content": "malformed",
            "post_at": "not-a-date",
        }
        future_entry = {
            "id": "future-1",
            "guild_id": "2",
            "channel_id": "20",
            "content": "later",
            "post_at": _future_iso(2),
        }
        self.mock_load.return_value = [bad_entry, future_entry]

        with patch.object(bot_module, "_send_daily_challenge", new=AsyncMock(return_value=True)):
            await bot_module._post_scheduled_challenges()

        saved = self.mock_save.call_args[0][0]
        saved_ids = {c["id"] for c in saved}
        self.assertNotIn("bad-post-at-2", saved_ids)
        self.assertIn("future-1", saved_ids)
        self.assertEqual(len(saved), 1)


class TestScheduledLoopSurvivesBadChannelId(unittest.IsolatedAsyncioTestCase):
    """Issue #75: a malformed channel_id already on disk (e.g. from a legacy
    entry, or one that slipped past validation) must not crash the loop --
    _send_daily_challenge must catch the ValueError from int() and return
    False, and _post_scheduled_challenges must continue on to other entries
    and to future ticks rather than propagating the exception."""

    async def asyncSetUp(self):
        self._load_patch = patch.object(bot_module, "_load_schedule")
        self._save_patch = patch.object(bot_module, "_save_schedule")
        self.mock_load = self._load_patch.start()
        self.mock_save = self._save_patch.start()

    async def asyncTearDown(self):
        self._load_patch.stop()
        self._save_patch.stop()

    async def test_send_daily_challenge_returns_false_for_non_numeric_channel_id(self):
        """A non-numeric channel_id must not raise ValueError out of
        _send_daily_challenge; it must be treated as a failed send."""
        result = await bot_module._send_daily_challenge(
            {"guild_id": "1", "channel_id": "#art-share", "day": "Day 1", "post_at": _past_iso()}
        )
        self.assertFalse(result)

    async def test_loop_survives_and_processes_other_guilds(self):
        """One entry with a malformed channel_id must not stop delivery to
        other, valid guilds in the same tick -- and must not raise."""
        bad_channel_entry = {
            "id": "bad-channel-1",
            "guild_id": "1",
            "channel_id": "not-a-number",
            "day": "Day 1",
            "post_at": _past_iso(),
        }
        good_channel = MagicMock()
        good_channel.create_thread = AsyncMock()

        self.mock_load.return_value = [bad_channel_entry]

        with patch.object(bot_module.bot, "get_channel", return_value=good_channel):
            # Must not raise -- this is the core regression check: previously
            # the ValueError from int("not-a-number") propagated out of
            # _send_daily_challenge, out of the for-loop, and out of
            # _post_scheduled_challenges, killing the tasks.loop forever.
            await bot_module._post_scheduled_challenges()

        # The bad entry failed (channel_id is not numeric) and is kept for retry,
        # not silently lost and not fatal to the tick.
        saved = self.mock_save.call_args[0][0]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["id"], "bad-channel-1")

    async def test_loop_can_run_again_after_a_bad_entry(self):
        """A second tick after a bad-channel_id entry still runs normally --
        proof the loop function itself never raised and is still callable."""
        bad_channel_entry = {
            "id": "bad-channel-2",
            "guild_id": "1",
            "channel_id": "totally-invalid",
            "day": "Day 1",
            "post_at": _past_iso(),
        }
        self.mock_load.return_value = [bad_channel_entry]

        # First tick: must not raise despite the bad channel_id.
        await bot_module._post_scheduled_challenges()

        # Second tick, now with a clean schedule -- proves the loop function
        # is still healthy and callable (i.e. it wasn't left in a broken state).
        self.mock_load.return_value = []
        await bot_module._post_scheduled_challenges()

        self.assertEqual(self.mock_save.call_count, 2)
        self.assertEqual(self.mock_save.call_args[0][0], [])


class TestScheduledLoopErrorHandlerRestart(unittest.IsolatedAsyncioTestCase):
    """PR #76 review (discussion_r3702675189): pycord invokes the loop's
    .error handler from *inside* the still-finishing loop task, so
    is_running() reports True at that point regardless of whether the loop
    is about to actually stop. Checking it synchronously in the handler (as
    the original fix did) always skips start(), so any exception that
    escapes the per-entry send guard still permanently kills scheduled
    delivery -- the exact failure mode #76 was supposed to close off.
    """

    async def asyncSetUp(self):
        self._sleep_patch = patch.object(bot_module.asyncio, "sleep", AsyncMock())
        self.mock_sleep = self._sleep_patch.start()

    async def asyncTearDown(self):
        self._sleep_patch.stop()

    async def test_error_handler_does_not_call_start_synchronously(self):
        """The handler itself must never call start() directly -- only
        schedule the deferred restart -- since is_running() is unreliable
        at the point the handler runs."""
        with patch.object(bot_module.asyncio, "create_task") as mock_create_task:
            with patch.object(bot_module._post_scheduled_challenges, "start") as mock_start:
                await bot_module._post_scheduled_challenges_error(RuntimeError("boom"))
                mock_start.assert_not_called()
                mock_create_task.assert_called_once()
                # Avoid a "coroutine was never awaited" warning: create_task
                # is mocked out, so the coroutine it was handed is never
                # actually scheduled/run.
                mock_create_task.call_args[0][0].close()

    async def test_restart_waits_for_is_running_to_go_false_before_starting(self):
        """is_running() reporting True for a few polls (simulating the loop
        task still finishing up) must not cause the restart to be skipped;
        it must retry until is_running() is False, then start() exactly
        once."""
        is_running_results = [True, True, False]
        with patch.object(
            bot_module._post_scheduled_challenges, "is_running",
            side_effect=is_running_results,
        ) as mock_is_running:
            with patch.object(bot_module._post_scheduled_challenges, "start") as mock_start:
                await bot_module._restart_scheduled_challenges_loop()

        self.assertEqual(mock_is_running.call_count, 3)
        mock_start.assert_called_once()

    async def test_restart_gives_up_if_is_running_never_goes_false(self):
        """A loop that never actually reports stopped must not retry
        forever or raise -- it should give up after a bounded number of
        attempts and never call start() (which would raise on a genuinely
        still-running loop)."""
        with patch.object(
            bot_module._post_scheduled_challenges, "is_running", return_value=True,
        ) as mock_is_running:
            with patch.object(bot_module._post_scheduled_challenges, "start") as mock_start:
                await bot_module._restart_scheduled_challenges_loop()

        self.assertEqual(
            mock_is_running.call_count, bot_module._LOOP_RESTART_POLL_MAX_ATTEMPTS
        )
        mock_start.assert_not_called()


class TestDailyChallengeChannelIdValidation(unittest.IsolatedAsyncioTestCase):
    """Issue #75: /daily_challenge must validate channel_id before scheduling.

    (a) Non-numeric / malformed channel_id is rejected at command time.
    (b) A numeric channel_id that doesn't resolve to a real, accessible
        channel is rejected.
    (c) A numeric channel_id that resolves to a channel in a *different*
        guild is rejected.
    (d) A valid channel_id in the same guild is accepted and scheduled.
    """

    def _make_ctx(self, guild_id=42):
        ctx = MagicMock(spec=_discord_stub.ApplicationContext)
        ctx.guild_id = guild_id
        ctx.defer = AsyncMock()
        ctx.followup = MagicMock()
        ctx.followup.send = AsyncMock()

        # These tests exercise channel_id validation, not the admin gate
        # added for #79 -- wire up a caller who passes _require_guild_admin
        # (a resolvable guild member with manage_guild) so that gate is a
        # no-op here.
        #
        # Use bot_module.discord.Member (not _discord_stub.Member) as the
        # spec: if another test module in the same pytest session imported
        # the real `discord` package first, sys.modules["discord"] is
        # already the real one by the time this file's setdefault() call
        # below runs, so bot_module.discord may end up being the real
        # package rather than this file's stub. The isinstance() checks in
        # _require_guild_admin must match whichever one bot.py actually got.
        admin_member = MagicMock(spec=bot_module.discord.Member)
        admin_member.id = 1
        admin_member.guild_permissions = MagicMock()
        admin_member.guild_permissions.administrator = False
        admin_member.guild_permissions.manage_guild = True

        guild = MagicMock()
        guild.id = guild_id
        guild.get_member.return_value = admin_member

        ctx.guild = guild
        ctx.author = admin_member
        ctx.response = MagicMock()
        ctx.response.is_done.return_value = False
        ctx.respond = AsyncMock()
        return ctx

    async def asyncSetUp(self):
        self._load_patch = patch.object(bot_module, "_load_schedule", return_value=[])
        self._save_patch = patch.object(bot_module, "_save_schedule")
        self.mock_load = self._load_patch.start()
        self.mock_save = self._save_patch.start()

    async def asyncTearDown(self):
        self._load_patch.stop()
        self._save_patch.stop()

    @staticmethod
    def _option_defaults():
        """Default values py-cord would normally supply for the optional
        Option-typed parameters. The stub's discord.Option(...) evaluates to
        None at import time (it's used purely as a type annotation, not a
        Python-level default), so calling the command function directly
        requires passing these explicitly."""
        return dict(
            release_time="18:00",
            release_date=None,
            reference=None,
            minimum_time=None,
            extra_challenge=None,
            description=None,
        )

    async def test_non_numeric_channel_id_is_rejected(self):
        ctx = self._make_ctx()

        await bot_module.daily_challenge(
            ctx, day="Day 1", channel_id="#art-share", **self._option_defaults(),
        )

        ctx.followup.send.assert_awaited_once()
        (msg,), kwargs = ctx.followup.send.call_args
        self.assertIn("not a valid channel ID", msg)
        self.assertTrue(kwargs.get("ephemeral"))
        # Must not have been scheduled.
        self.mock_save.assert_not_called()

    async def test_unresolvable_channel_id_is_rejected(self):
        ctx = self._make_ctx()

        with patch.object(bot_module.bot, "get_channel", return_value=None):
            await bot_module.daily_challenge(
                ctx, day="Day 1", channel_id="999999999999999999", **self._option_defaults(),
            )

        ctx.followup.send.assert_awaited_once()
        (msg,), kwargs = ctx.followup.send.call_args
        self.assertIn("can't see a channel", msg)
        self.mock_save.assert_not_called()

    async def test_channel_id_in_different_guild_is_rejected(self):
        ctx = self._make_ctx(guild_id=42)

        other_guild = MagicMock()
        other_guild.id = 999
        foreign_channel = MagicMock()
        foreign_channel.guild = other_guild

        with patch.object(bot_module.bot, "get_channel", return_value=foreign_channel):
            await bot_module.daily_challenge(
                ctx, day="Day 1", channel_id="123", **self._option_defaults(),
            )

        ctx.followup.send.assert_awaited_once()
        (msg,), kwargs = ctx.followup.send.call_args
        self.assertIn("doesn't belong to this server", msg)
        self.mock_save.assert_not_called()

    async def test_valid_channel_id_in_same_guild_is_accepted(self):
        ctx = self._make_ctx(guild_id=42)

        same_guild = MagicMock()
        same_guild.id = 42
        valid_channel = MagicMock()
        valid_channel.guild = same_guild

        with patch.object(bot_module.bot, "get_channel", return_value=valid_channel), \
             patch.object(bot_module, "_load_references", return_value=[]), \
             patch.object(bot_module, "_parse_release_datetime", return_value=_future_iso(1)):
            await bot_module.daily_challenge(
                ctx, day="Day 1", channel_id="123", **self._option_defaults(),
            )

        self.mock_save.assert_called_once()
        saved = self.mock_save.call_args[0][0]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["channel_id"], "123")
        # Success message, not an error.
        ctx.followup.send.assert_awaited_once()
        (msg,), _ = ctx.followup.send.call_args
        self.assertIn("scheduled", msg)


if __name__ == "__main__":
    unittest.main()
