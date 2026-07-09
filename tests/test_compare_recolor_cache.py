"""Regression tests for #3: /compare and /recolor must consult the image
result cache instead of re-running KMeans on every invocation, even when the
same image bytes are supplied again (e.g. refining a recolor against
different targets, or comparing one image against several others).
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import bot
from conftest import make_solid_image


class _FakeAttachment:
    """Minimal stand-in for discord.Attachment with an async .read()."""

    def __init__(self, data: bytes, filename="art.png"):
        self._data = data
        self.content_type = "image/png"
        self.size = len(data)
        self.width = 32
        self.height = 32
        self.filename = filename

    async def read(self) -> bytes:
        return self._data


def _make_ctx(user_id: int = 1):
    ctx = MagicMock()
    ctx.author.id = user_id
    ctx.defer = AsyncMock()
    ctx.followup = MagicMock()
    ctx.followup.send = AsyncMock()
    return ctx


@pytest.fixture(autouse=True)
def _clear_state():
    bot._IMAGE_CACHE.clear()
    bot._USER_COOLDOWNS.clear()
    yield
    bot._IMAGE_CACHE.clear()
    bot._USER_COOLDOWNS.clear()


class TestCompareCmdUsesCache:
    @pytest.mark.asyncio
    async def test_reuploading_same_image_skips_kmeans(self, png_bytes):
        data_a = png_bytes(make_solid_image((200, 40, 40)))
        data_b = png_bytes(make_solid_image((40, 200, 40)))
        image_a = _FakeAttachment(data_a, "a.png")
        image_b = _FakeAttachment(data_b, "b.png")

        real_extract = bot.extract_dominant_colors
        with patch("bot.extract_dominant_colors", wraps=real_extract) as spy:
            ctx1 = _make_ctx(user_id=1)
            await bot.compare_cmd.callback(ctx1, image_a, image_b, num_colors=4)
            assert spy.call_count == 2  # first upload: cache miss for both images

            ctx2 = _make_ctx(user_id=2)  # different user avoids per-user cooldown
            await bot.compare_cmd.callback(ctx2, image_a, image_b, num_colors=4)
            assert spy.call_count == 2, (
                "Re-comparing the same two images should hit the cache and "
                "not call extract_dominant_colors again"
            )


class TestRecolorCmdUsesCache:
    @pytest.mark.asyncio
    async def test_reusing_same_source_skips_kmeans(self, png_bytes):
        src_data = png_bytes(make_solid_image((200, 40, 40)))
        source = _FakeAttachment(src_data, "source.png")
        target1 = _FakeAttachment(png_bytes(make_solid_image((10, 10, 10))), "target1.png")
        target2 = _FakeAttachment(png_bytes(make_solid_image((250, 250, 250))), "target2.png")

        real_extract = bot.extract_dominant_colors
        with patch("bot.extract_dominant_colors", wraps=real_extract) as spy:
            ctx1 = _make_ctx(user_id=1)
            await bot.recolor_cmd.callback(ctx1, source, target1, num_colors=4)
            assert spy.call_count == 1  # first upload of this source: cache miss

            ctx2 = _make_ctx(user_id=2)  # different user avoids per-user cooldown
            await bot.recolor_cmd.callback(ctx2, source, target2, num_colors=4)
            assert spy.call_count == 1, (
                "Recoloring against the same source image with a different "
                "target should hit the cache and not re-run KMeans on the source"
            )
