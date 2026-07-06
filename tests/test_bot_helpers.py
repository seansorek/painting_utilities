"""Tests for the pure helpers and in-memory cache in bot.py.

Importing ``bot`` is safe: ``discord.Bot(...)`` is constructed at import but
never connects, and ``bot.run()`` only executes under ``__main__``.
"""
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

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

    def test_cache_set_does_not_raise_on_concurrent_eviction(self):
        """_cache_set must not raise if another coroutine already evicted the oldest entry."""
        # Fill cache to max
        for i in range(bot._CACHE_MAX):
            bot._cache_set(f"img{i}".encode(), 8, 0.0, 0.0, i, i, i)

        # Simulate the race: manually evict the first key before _cache_set runs the eviction
        first_key = next(iter(bot._IMAGE_CACHE))
        del bot._IMAGE_CACHE[first_key]

        # _cache_set should not raise even though the oldest key is already gone
        bot._cache_set(b"race_winner", 8, 0.0, 0.0, "x", "y", "z")
        assert bot._cache_get(b"race_winner", 8) == ("x", "y", "z")

    def test_cache_set_does_not_raise_on_empty_cache_race(self):
        """_cache_set must not raise if another coroutine emptied the cache entirely."""
        # Fill cache to max
        for i in range(bot._CACHE_MAX):
            bot._cache_set(f"img{i}".encode(), 8, 0.0, 0.0, i, i, i)

        # Simulate extreme race: entire cache was cleared between len() check and pop()
        bot._IMAGE_CACHE.clear()

        # _cache_set should not raise on StopIteration from next(iter({}))
        bot._cache_set(b"race_empty", 8, 0.0, 0.0, "a", "b", "c")
        assert bot._cache_get(b"race_empty", 8) == ("a", "b", "c")


class TestOnReadyGuard:
    """Verify that on_ready guards _post_scheduled_challenges.start() with is_running()."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)


class TestScheduleAndReferencesOffThread:
    """_load_schedule/_save_schedule/_load_references do blocking file I/O and
    must be safe to call via asyncio.to_thread from async handlers (#49)."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_save_then_load_schedule_round_trips_via_to_thread(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot, "_SCHEDULE_FILE", str(tmp_path / "daily_schedule.json"))
        challenges = [{"id": "abc", "day": "Day 1", "post_at": "2026-01-01T00:00:00+00:00"}]

        self._run(asyncio.to_thread(bot._save_schedule, challenges))
        loaded = self._run(asyncio.to_thread(bot._load_schedule))

        assert loaded == challenges

    def test_load_schedule_missing_file_returns_empty_list_via_to_thread(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot, "_SCHEDULE_FILE", str(tmp_path / "does_not_exist.json"))

        assert self._run(asyncio.to_thread(bot._load_schedule)) == []

    def test_load_references_via_to_thread(self, tmp_path, monkeypatch):
        refs_path = tmp_path / "references.json"
        refs_path.write_text('["http://example.com/a.png"]', encoding="utf-8")
        monkeypatch.setattr(bot, "_REFERENCES_FILE", str(refs_path))

        assert self._run(asyncio.to_thread(bot._load_references)) == ["http://example.com/a.png"]

    def _patch_bot_user(self):
        """Return a context manager that makes bot.bot.user a non-None mock.

        discord.Bot.user is a property backed by _connection._state.user, so we
        patch the underlying attribute directly on the live instance via __dict__
        of the class, using patch.object with new_callable=PropertyMock.
        """
        from unittest.mock import PropertyMock
        mock_user = MagicMock()
        mock_user.id = 123
        return patch.object(type(bot.bot), "user", new_callable=PropertyMock, return_value=mock_user)

    def test_start_not_called_when_already_running(self):
        """If the loop reports is_running() == True, start() must NOT be called."""
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True

        with patch.object(bot, "_post_scheduled_challenges", mock_loop):
            with self._patch_bot_user():
                self._run(bot.on_ready())

        mock_loop.is_running.assert_called_once()
        mock_loop.start.assert_not_called()

    def test_start_called_when_not_running(self):
        """If the loop reports is_running() == False, start() must be called exactly once."""
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False

        with patch.object(bot, "_post_scheduled_challenges", mock_loop):
            with self._patch_bot_user():
                self._run(bot.on_ready())

        mock_loop.is_running.assert_called_once()
        mock_loop.start.assert_called_once()


class TestSanitizeFilenameComponent:
    """`_sanitize_filename_component` guards the .gpl attachment filename
    built from the user-supplied `palette_name` (issue #32): control
    characters and path separators must not survive into the filename.
    """

    def test_strips_newlines_and_carriage_returns(self):
        result = bot._sanitize_filename_component("foo\nColumns: 99\n255 0 0 evil")
        assert "\n" not in result
        assert result == "fooColumns_ 99255 0 0 evil"

        assert bot._sanitize_filename_component("foo\r\nbar") == "foobar"

    def test_replaces_path_separators(self):
        result = bot._sanitize_filename_component("../../etc/passwd")
        assert "/" not in result
        assert result == "_.._etc_passwd"

        result2 = bot._sanitize_filename_component("C:\\Windows\\System32")
        assert "\\" not in result2
        assert ":" not in result2
        assert result2 == "C__Windows_System32"

    def test_replaces_other_filesystem_unsafe_chars(self):
        result = bot._sanitize_filename_component('a*b?c"d<e>f|g')
        for ch in '*?"<>|':
            assert ch not in result
        assert result == "a_b_c_d_e_f_g"

    def test_full_filename_has_no_injected_or_unsafe_chars(self):
        malicious = "foo\nColumns: 99\n255 0 0 evil/../secret"
        safe = bot._sanitize_filename_component(malicious)
        filename = f"{safe}.gpl"
        for ch in "\n\r/\\":
            assert ch not in filename

    def test_normal_name_unchanged(self):
        assert bot._sanitize_filename_component("My Palette") == "My Palette"

    def test_empty_or_dots_only_falls_back_to_default(self):
        assert bot._sanitize_filename_component("") == "Palette"
        assert bot._sanitize_filename_component("...") == "Palette"
        assert bot._sanitize_filename_component("\n\r") == "Palette"
