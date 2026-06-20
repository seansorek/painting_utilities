"""Tests for the per-user cooldown mechanism in bot.py.

Verifies that:
- A rejected upload (non-image file) does NOT consume the cooldown.
- A successful validation DOES consume the cooldown.
- The cooldown check correctly blocks requests within the cooldown window.
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import bot


class _FakeAttachment:
    """Minimal stand-in for discord.Attachment."""

    def __init__(self, content_type="image/png", size=1000, width=100, height=100,
                 filename="art.png"):
        self.content_type = content_type
        self.size = size
        self.width = width
        self.height = height
        self.filename = filename


def _make_ctx(user_id: int = 42, command_name: str = "analyze"):
    """Build a minimal mock ApplicationContext for cooldown tests."""
    ctx = MagicMock()
    ctx.author.id = user_id
    ctx.command.name = command_name
    ctx.followup = MagicMock()
    ctx.followup.send = AsyncMock()
    return ctx


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestCooldownNotConsumedOnRejection:
    """A rejected upload must NOT lock the user out of the next request."""

    @pytest.fixture(autouse=True)
    def _clear_cooldowns(self):
        bot._USER_COOLDOWNS.clear()
        yield
        bot._USER_COOLDOWNS.clear()

    def test_rejected_image_does_not_stamp_cooldown(self):
        """_validate_image rejects a non-image file; user should NOT be in cooldown."""
        ctx = _make_ctx(user_id=1)
        bad_attachment = _FakeAttachment(content_type="text/plain", filename="notes.txt")

        # Simulate the validation rejection
        result = _run(bot._validate_image(ctx, bad_attachment))
        assert result is False, "_validate_image should return False for non-image"

        # The user should NOT have a cooldown entry
        assert 1 not in bot._USER_COOLDOWNS

    def test_rejected_then_valid_request_is_allowed(self):
        """After a rejected upload, the next request must pass the cooldown check."""
        user_id = 2

        # Step 1: user triggers a heavy command — cooldown check passes (no prior stamp)
        ctx1 = _make_ctx(user_id=user_id)
        result = _run(bot._cooldown_check(ctx1))
        assert result is True

        # Step 2: validation rejects the upload — no _consume_cooldown call
        bad_attachment = _FakeAttachment(content_type="application/pdf", filename="doc.pdf")
        valid = _run(bot._validate_image(ctx1, bad_attachment))
        assert valid is False

        # Step 3: user immediately retries with a valid image — cooldown check should pass
        ctx2 = _make_ctx(user_id=user_id)
        result2 = _run(bot._cooldown_check(ctx2))
        assert result2 is True, "Cooldown check should pass because no cooldown was consumed"

    def test_oversized_file_does_not_stamp_cooldown(self):
        """An oversized file rejected by _validate_image must not consume cooldown."""
        ctx = _make_ctx(user_id=3)
        big_attachment = _FakeAttachment(size=bot.MAX_FILE_BYTES + 1)

        result = _run(bot._validate_image(ctx, big_attachment))
        assert result is False

        assert 3 not in bot._USER_COOLDOWNS


class TestCooldownConsumedOnSuccess:
    """A successful validation MUST consume the cooldown."""

    @pytest.fixture(autouse=True)
    def _clear_cooldowns(self):
        bot._USER_COOLDOWNS.clear()
        yield
        bot._USER_COOLDOWNS.clear()

    def test_consume_cooldown_stamps_user(self):
        """_consume_cooldown sets the timestamp for the given user."""
        user_id = 10
        assert user_id not in bot._USER_COOLDOWNS

        bot._consume_cooldown(user_id)

        assert user_id in bot._USER_COOLDOWNS
        assert bot._USER_COOLDOWNS[user_id] > 0

    def test_consume_then_check_raises_cooldown_error(self):
        """After _consume_cooldown, the next cooldown check within the window must raise."""
        user_id = 11

        # Simulate: validation passed, cooldown consumed
        bot._consume_cooldown(user_id)

        # Immediately try another heavy command — should be blocked
        ctx = _make_ctx(user_id=user_id)
        with pytest.raises(bot._CooldownError) as exc_info:
            _run(bot._cooldown_check(ctx))
        assert exc_info.value.retry_after > 0

    def test_cooldown_expires_after_window(self):
        """After the cooldown window elapses, the user can run again."""
        user_id = 12

        # Stamp a cooldown far enough in the past
        bot._USER_COOLDOWNS[user_id] = time.monotonic() - bot._COOLDOWN_SECONDS - 1

        ctx = _make_ctx(user_id=user_id)
        result = _run(bot._cooldown_check(ctx))
        assert result is True


class TestCooldownCheckDoesNotStamp:
    """The _cooldown_check function must NOT stamp the cooldown itself."""

    @pytest.fixture(autouse=True)
    def _clear_cooldowns(self):
        bot._USER_COOLDOWNS.clear()
        yield
        bot._USER_COOLDOWNS.clear()

    def test_check_alone_does_not_create_entry(self):
        """Passing _cooldown_check must not add the user to _USER_COOLDOWNS."""
        user_id = 20
        ctx = _make_ctx(user_id=user_id)

        result = _run(bot._cooldown_check(ctx))
        assert result is True
        assert user_id not in bot._USER_COOLDOWNS

    def test_two_checks_without_consume_both_pass(self):
        """Without _consume_cooldown, consecutive checks must both pass."""
        user_id = 21
        ctx1 = _make_ctx(user_id=user_id)
        ctx2 = _make_ctx(user_id=user_id)

        result1 = _run(bot._cooldown_check(ctx1))
        result2 = _run(bot._cooldown_check(ctx2))

        assert result1 is True
        assert result2 is True

    def test_non_heavy_command_skips_check(self):
        """A non-heavy command like 'help' should bypass the cooldown entirely."""
        user_id = 22
        ctx = _make_ctx(user_id=user_id, command_name="help")

        result = _run(bot._cooldown_check(ctx))
        assert result is True
        assert user_id not in bot._USER_COOLDOWNS
