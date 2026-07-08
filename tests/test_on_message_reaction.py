"""Tests for the on_message star-reaction handler in bot.py.

Covers the fix for #28: message.add_reaction can raise discord.Forbidden
(missing Add Reactions / Read Message History perms), discord.NotFound
(message deleted before we react), or a transient discord.HTTPException.
Since discord.Forbidden and discord.NotFound both subclass HTTPException,
on_message must catch HTTPException around the add_reaction call so a
misconfigured guild doesn't produce an unhandled traceback per image post.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

import bot


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeAttachment:
    """Minimal stand-in for discord.Attachment."""

    def __init__(self, content_type="image/png"):
        self.content_type = content_type


def _make_message(guild_id: int = 1, channel_id: int = 55, is_bot: bool = False):
    """Build a mock discord.Message posted with an image in a tracked thread."""
    guild = MagicMock(spec=discord.Guild)
    guild.id = guild_id

    channel = MagicMock(spec=discord.Thread)
    channel.parent_id = channel_id

    message = MagicMock(spec=discord.Message)
    message.author = MagicMock()
    message.author.bot = is_bot
    message.guild = guild
    message.channel = channel
    message.attachments = [_FakeAttachment("image/png")]
    message.add_reaction = AsyncMock()

    return message


class TestOnMessageReactionErrorHandling:
    """add_reaction failures must not propagate out of on_message."""

    def test_forbidden_does_not_propagate(self):
        message = _make_message()
        message.add_reaction.side_effect = discord.Forbidden(
            MagicMock(status=403), "Missing Permissions"
        )
        with patch.object(bot, "_get_guild_channel", return_value="55"):
            _run(bot.on_message(message))  # should not raise
        message.add_reaction.assert_awaited_once_with("⭐")

    def test_not_found_does_not_propagate(self):
        message = _make_message()
        message.add_reaction.side_effect = discord.NotFound(
            MagicMock(status=404), "Unknown Message"
        )
        with patch.object(bot, "_get_guild_channel", return_value="55"):
            _run(bot.on_message(message))  # should not raise

    def test_generic_http_exception_does_not_propagate(self):
        message = _make_message()
        message.add_reaction.side_effect = discord.HTTPException(
            MagicMock(status=500), "Internal Server Error"
        )
        with patch.object(bot, "_get_guild_channel", return_value="55"):
            _run(bot.on_message(message))  # should not raise

    def test_successful_reaction_still_added(self):
        message = _make_message()
        with patch.object(bot, "_get_guild_channel", return_value="55"):
            _run(bot.on_message(message))
        message.add_reaction.assert_awaited_once_with("⭐")
