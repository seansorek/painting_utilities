"""Tests for the _require_bot_role global check in bot.py.

Covers the fix for #45: when no required role is configured for a guild,
all users should be allowed through (not blocked).
"""
import asyncio
from unittest.mock import MagicMock, patch

import discord
import pytest

import bot


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_role(role_id: int) -> MagicMock:
    role = MagicMock(spec=discord.Role)
    role.id = role_id
    return role


def _make_member(
    user_id: int = 42,
    guild_id: int = 1,
    is_admin: bool = False,
    is_owner: bool = False,
    role_ids: list[int] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Return (ctx, member) mocks wired together for _require_bot_role."""
    guild = MagicMock(spec=discord.Guild)
    guild.guild_id = guild_id
    guild.owner_id = user_id if is_owner else 0

    member = MagicMock(spec=discord.Member)
    member.id = user_id
    member.guild_permissions = MagicMock()
    member.guild_permissions.administrator = is_admin
    member.guild_permissions.manage_guild = False
    member.roles = [_make_role(rid) for rid in (role_ids or [])]

    guild.get_member.return_value = member

    ctx = MagicMock(spec=discord.ApplicationContext)
    ctx.guild = guild
    ctx.guild_id = guild_id
    ctx.author = member
    ctx.author.id = user_id

    return ctx, member


class TestRequireBotRoleNoRoleConfigured:
    """When no required role is set for a guild, all users should pass."""

    def test_regular_user_allowed_when_no_role_configured(self):
        ctx, _ = _make_member(user_id=100, is_admin=False)
        with patch.object(bot, "_get_guild_required_role", return_value=None):
            result = _run(bot._require_bot_role(ctx))
        assert result is True

    def test_regular_user_allowed_when_role_is_empty_string(self):
        ctx, _ = _make_member(user_id=101, is_admin=False)
        with patch.object(bot, "_get_guild_required_role", return_value=""):
            result = _run(bot._require_bot_role(ctx))
        assert result is True


class TestRequireBotRoleWithRoleConfigured:
    """When a required role IS set, only users with that role (or admins) pass."""

    def test_user_with_required_role_allowed(self):
        ctx, _ = _make_member(user_id=200, role_ids=[999])
        with patch.object(bot, "_get_guild_required_role", return_value="999"):
            result = _run(bot._require_bot_role(ctx))
        assert result is True

    def test_user_without_required_role_blocked(self):
        ctx, _ = _make_member(user_id=201, role_ids=[111])
        with patch.object(bot, "_get_guild_required_role", return_value="999"):
            result = _run(bot._require_bot_role(ctx))
        assert result is False

    def test_admin_bypasses_role_check(self):
        ctx, _ = _make_member(user_id=202, is_admin=True)
        with patch.object(bot, "_get_guild_required_role", return_value="999"):
            result = _run(bot._require_bot_role(ctx))
        assert result is True

    def test_owner_bypasses_role_check(self):
        ctx, _ = _make_member(user_id=203, is_owner=True)
        with patch.object(bot, "_get_guild_required_role", return_value="999"):
            result = _run(bot._require_bot_role(ctx))
        assert result is True


class TestRequireBotRoleDM:
    """DM context (no guild) should always be allowed."""

    def test_dm_context_allowed(self):
        ctx = MagicMock(spec=discord.ApplicationContext)
        ctx.guild = None
        result = _run(bot._require_bot_role(ctx))
        assert result is True
