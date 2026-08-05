"""Tests for the runtime manage_guild admin guard added for #79.

Issue #79: the schedule/configuration commands were marked with
``@discord.default_permissions(manage_guild=True)`` but performed no runtime
check. That decorator only sets Discord's *default* command-permission UI --
a guild admin can override it in Discord to expose the command to any role
or member. Without a runtime check, an exposed command would execute for a
non-admin caller.

These tests prove:
  1. ``_require_guild_admin`` itself correctly allows/rejects based on the
     caller's live ``guild_permissions.manage_guild`` / ``.administrator``.
  2. Every admin command identified in #79 rejects a non-admin caller before
     doing any state-mutating work -- even though nothing else in the test
     setup would have stopped the command from succeeding.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

import bot


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_member(is_admin: bool = False, has_manage_guild: bool = False) -> MagicMock:
    member = MagicMock(spec=discord.Member)
    member.id = 42
    member.guild_permissions = MagicMock()
    member.guild_permissions.administrator = is_admin
    member.guild_permissions.manage_guild = has_manage_guild
    return member


def _make_ctx(member: MagicMock, guild_id: int = 1) -> MagicMock:
    """Build a mock ApplicationContext wired for _require_guild_admin.

    ``ctx.response.is_done()`` defaults to False so the rejection path uses
    ``ctx.respond`` (matching a command that has not deferred yet); tests
    for commands that defer first override this to True.
    """
    # Not spec'd to discord.Guild: this repo's test_schedule.py installs a
    # minimal discord stub (lacking a Guild class) into sys.modules for the
    # duration of the pytest session when it runs before this file, and the
    # only interface this test needs from `guild` is get_member().
    guild = MagicMock()
    guild.get_member.return_value = member

    ctx = MagicMock(spec=discord.ApplicationContext)
    ctx.guild = guild
    ctx.guild_id = guild_id
    ctx.author = member
    ctx.response = MagicMock()
    ctx.response.is_done.return_value = False
    ctx.respond = AsyncMock()
    ctx.followup = MagicMock()
    ctx.followup.send = AsyncMock()
    ctx.defer = AsyncMock()
    return ctx


class TestRequireGuildAdminHelper:
    """Direct tests of the _require_guild_admin runtime guard."""

    def test_member_with_manage_guild_is_allowed(self):
        member = _make_member(has_manage_guild=True)
        ctx = _make_ctx(member)
        result = _run(bot._require_guild_admin(ctx))
        assert result is True
        ctx.respond.assert_not_awaited()

    def test_member_with_administrator_is_allowed(self):
        member = _make_member(is_admin=True)
        ctx = _make_ctx(member)
        result = _run(bot._require_guild_admin(ctx))
        assert result is True
        ctx.respond.assert_not_awaited()

    def test_member_without_manage_guild_is_rejected(self):
        """A caller with neither administrator nor manage_guild must be
        rejected even though this is precisely the scenario #79 describes:
        the command was exposed to them via a Discord permission override,
        and the global _require_bot_role check (not exercised here) would
        have let them through."""
        member = _make_member(is_admin=False, has_manage_guild=False)
        ctx = _make_ctx(member)
        result = _run(bot._require_guild_admin(ctx))
        assert result is False
        ctx.respond.assert_awaited_once()
        (msg,), kwargs = ctx.respond.call_args
        assert "permission" in msg.lower()
        assert kwargs.get("ephemeral") is True

    def test_rejection_uses_followup_when_already_deferred(self):
        member = _make_member(is_admin=False, has_manage_guild=False)
        ctx = _make_ctx(member)
        ctx.response.is_done.return_value = True
        result = _run(bot._require_guild_admin(ctx))
        assert result is False
        ctx.followup.send.assert_awaited_once()
        ctx.respond.assert_not_awaited()

    def test_unresolvable_member_is_rejected(self):
        """If the guild can't resolve a live Member for the caller (e.g. a
        stale cache), fail closed rather than assuming authorization."""
        guild = MagicMock()
        guild.get_member.return_value = None

        ctx = MagicMock(spec=discord.ApplicationContext)
        ctx.guild = guild
        ctx.guild_id = 1
        ctx.author = MagicMock()  # not a discord.Member
        ctx.response = MagicMock()
        ctx.response.is_done.return_value = False
        ctx.respond = AsyncMock()

        result = _run(bot._require_guild_admin(ctx))
        assert result is False
        ctx.respond.assert_awaited_once()


class TestAdminCommandsRejectNonAdmin:
    """Each command identified in #79 must reject a non-admin caller before
    performing any state-mutating work, regardless of what
    @discord.default_permissions(manage_guild=True) (an overridable default,
    not a runtime guarantee) would otherwise allow through."""

    def _non_admin_ctx(self, guild_id: int = 1) -> MagicMock:
        member = _make_member(is_admin=False, has_manage_guild=False)
        ctx = _make_ctx(member, guild_id=guild_id)
        ctx.response.is_done.return_value = True  # these commands defer/respond first
        return ctx

    def test_list_schedule_rejects_non_admin(self):
        ctx = self._non_admin_ctx()
        with patch.object(bot, "_load_schedule") as mock_load:
            _run(bot.list_schedule(ctx))
            mock_load.assert_not_called()
        ctx.followup.send.assert_awaited_once()
        (msg,), kwargs = ctx.followup.send.call_args
        assert "permission" in msg.lower()
        assert kwargs.get("ephemeral") is True

    def test_delete_challenge_rejects_non_admin(self):
        ctx = self._non_admin_ctx()
        with patch.object(bot, "_load_schedule") as mock_load, \
             patch.object(bot, "_save_schedule") as mock_save:
            _run(bot.delete_challenge(ctx, challenge="some-id"))
            mock_load.assert_not_called()
            mock_save.assert_not_called()
        ctx.followup.send.assert_awaited_once()

    def test_edit_challenge_rejects_non_admin(self):
        ctx = self._non_admin_ctx()
        with patch.object(bot, "_load_schedule") as mock_load, \
             patch.object(bot, "_save_schedule") as mock_save:
            _run(bot.edit_challenge(
                ctx, challenge="some-id", new_day="Day 99",
                description=None, release_time=None, reference=None,
                minimum_time=None, extra_challenge=None,
            ))
            mock_load.assert_not_called()
            mock_save.assert_not_called()
        ctx.followup.send.assert_awaited_once()

    def test_set_daily_channel_rejects_non_admin(self):
        ctx = self._non_admin_ctx()
        ctx.response.is_done.return_value = False  # this command never defers
        channel = MagicMock(spec=discord.ForumChannel)
        channel.id = 123
        with patch.object(bot, "_set_guild_channel", new=AsyncMock()) as mock_set:
            _run(bot.set_daily_channel(ctx, channel=channel))
            mock_set.assert_not_called()
        ctx.respond.assert_awaited_once()
        (msg,), kwargs = ctx.respond.call_args
        assert "permission" in msg.lower()
        assert kwargs.get("ephemeral") is True

    def test_set_daily_role_rejects_non_admin(self):
        ctx = self._non_admin_ctx()
        ctx.response.is_done.return_value = False
        role = MagicMock(spec=discord.Role)
        role.id = 999  # deliberately != ctx.guild_id, would otherwise pass the @everyone check
        with patch.object(bot, "_set_guild_daily_role", new=AsyncMock()) as mock_set:
            _run(bot.set_daily_role(ctx, role=role))
            mock_set.assert_not_called()
        ctx.respond.assert_awaited_once()

    def test_set_required_role_rejects_non_admin(self):
        ctx = self._non_admin_ctx()
        ctx.response.is_done.return_value = False
        role = MagicMock(spec=discord.Role)
        role.id = 999
        with patch.object(bot, "_set_guild_required_role", new=AsyncMock()) as mock_set:
            _run(bot.set_required_role(ctx, role=role))
            mock_set.assert_not_called()
        ctx.respond.assert_awaited_once()

    def test_daily_challenge_rejects_non_admin(self):
        ctx = self._non_admin_ctx()
        with patch.object(bot, "_load_schedule") as mock_load, \
             patch.object(bot, "_save_schedule") as mock_save:
            _run(bot.daily_challenge(
                ctx, day="Day 1", release_time="18:00", release_date=None,
                reference=None, minimum_time=None, extra_challenge=None,
                description=None, channel_id=None,
            ))
            mock_load.assert_not_called()
            mock_save.assert_not_called()
        ctx.followup.send.assert_awaited_once()
        (msg,), kwargs = ctx.followup.send.call_args
        assert "permission" in msg.lower()


if __name__ == "__main__":
    pytest.main([__file__])
