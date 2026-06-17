"""Tests for the pure helpers and in-memory cache in bot.py.

Importing ``bot`` is safe: ``discord.Bot(...)`` is constructed at import but
never connects, and ``bot.run()`` only executes under ``__main__``.
"""
import pytest

import bot


class TestPctBar:
    def test_full(self):
        assert bot._pct_bar(100, width=8) == "█" * 8

    def test_empty(self):
        assert bot._pct_bar(0, width=8) == "░" * 8

    def test_half(self):
        bar = bot._pct_bar(50, width=8)
        assert bar.count("█") == 4
        assert bar.count("░") == 4

    def test_total_length_is_width(self):
        for pct in (0, 12.5, 37, 50, 99, 100):
            assert len(bot._pct_bar(pct, width=10)) == 10


class TestHueRangeLabel:
    @pytest.mark.parametrize("hue_range, expected_name", [
        ((0, 30), "Red"),
        ((30, 60), "Orange"),
        ((60, 90), "Yellow"),
        ((90, 150), "Green"),
        ((210, 270), "Blue"),
    ])
    def test_named_ranges(self, hue_range, expected_name):
        assert bot._hue_range_label(hue_range).startswith(expected_name)

    def test_includes_degrees(self):
        assert "0–30°" in bot._hue_range_label((0, 30))


class TestContrastLevel:
    @pytest.mark.parametrize("value, label", [
        (10, "Low"),
        (29.9, "Low"),
        (30, "Medium"),
        (69.9, "Medium"),
        (70, "High"),
        (120, "High"),
    ])
    def test_thresholds(self, value, label):
        assert bot._contrast_level(value).startswith(label)


class TestSaturationLevel:
    @pytest.mark.parametrize("value, label", [
        (10, "Low"),
        (19.9, "Low"),
        (20, "Medium"),
        (59.9, "Medium"),
        (60, "High"),
        (100, "High"),
    ])
    def test_thresholds(self, value, label):
        assert bot._saturation_level(value).startswith(label)


class TestColorLine:
    def test_basic_line(self):
        line = bot._color_line((255, 0, 0), 50, 100, show_rgb=False, show_cmyk=False)
        assert "#FF0000" in line
        assert "Red" in line
        assert "50.0%" in line

    def test_show_rgb(self):
        line = bot._color_line((255, 0, 0), 50, 100, show_rgb=True, show_cmyk=False)
        assert "RGB(255, 0, 0)" in line

    def test_show_cmyk(self):
        line = bot._color_line((255, 0, 0), 50, 100, show_rgb=False, show_cmyk=True)
        assert "CMYK(0%, 100%, 100%, 0%)" in line


class _FakeAttachment:
    def __init__(self, content_type="image/png", size=1000, width=100, height=100,
                 filename="art.png"):
        self.content_type = content_type
        self.size = size
        self.width = width
        self.height = height
        self.filename = filename


class TestImageRejectionReason:
    def test_valid_image_accepted(self):
        assert bot._image_rejection_reason(_FakeAttachment()) is None

    def test_non_image_rejected(self):
        reason = bot._image_rejection_reason(_FakeAttachment(content_type="text/plain"))
        assert reason and "not a valid image" in reason

    def test_oversized_bytes_rejected(self):
        reason = bot._image_rejection_reason(
            _FakeAttachment(size=bot.MAX_FILE_BYTES + 1)
        )
        assert reason and "too large" in reason

    def test_too_many_pixels_rejected(self):
        big = bot.MAX_IMAGE_PIXELS  # width*height will exceed the cap
        reason = bot._image_rejection_reason(
            _FakeAttachment(width=big, height=2)
        )
        assert reason and "too many pixels" in reason

    def test_missing_dimensions_does_not_crash(self):
        # Discord may omit width/height; that must not raise.
        assert bot._image_rejection_reason(
            _FakeAttachment(width=None, height=None)
        ) is None


class TestCache:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        bot._IMAGE_CACHE.clear()
        yield
        bot._IMAGE_CACHE.clear()

    def test_key_is_deterministic(self):
        data = b"abc"
        assert bot._cache_key(data, 5, 0.1, 0.2) == bot._cache_key(data, 5, 0.1, 0.2)

    def test_key_varies_with_params(self):
        data = b"abc"
        assert bot._cache_key(data, 5) != bot._cache_key(data, 6)
        assert bot._cache_key(data, 5, 0.1) != bot._cache_key(data, 5, 0.2)

    def test_set_then_get_roundtrip(self):
        data = b"img"
        bot._cache_set(data, 8, 0.0, 0.0, "colors", "counts", "stats")
        assert bot._cache_get(data, 8, 0.0, 0.0) == ("colors", "counts", "stats")

    def test_miss_returns_none(self):
        assert bot._cache_get(b"missing", 8) is None

    def test_lru_eviction_at_max(self):
        for i in range(bot._CACHE_MAX):
            bot._cache_set(f"img{i}".encode(), 8, 0.0, 0.0, i, i, i)
        assert len(bot._IMAGE_CACHE) == bot._CACHE_MAX
        # Oldest entry should still be present until one more is added.
        assert bot._cache_get(b"img0", 8) is not None
        bot._cache_set(b"overflow", 8, 0.0, 0.0, "x", "y", "z")
        assert len(bot._IMAGE_CACHE) == bot._CACHE_MAX
        # The first-inserted entry was evicted.
        assert bot._cache_get(b"img0", 8) is None
