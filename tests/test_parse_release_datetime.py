"""Tests for _parse_release_datetime -- date/time parsing with timezone handling.

Covers all supported date formats, year-rollover logic, the date_str=None
today/tomorrow fallback, and invalid-input error paths.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytz
import pytest

import bot

ET = pytz.timezone("America/New_York")


def _mock_now(year, month, day, hour=12, minute=0):
    """Return an aware ET datetime for use as a frozen 'now'."""
    return ET.localize(datetime(year, month, day, hour, minute, 0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_with_frozen_now(time_str, date_str=None, *, now=None):
    """Call _parse_release_datetime with a mocked datetime.now."""
    if now is None:
        now = _mock_now(2026, 6, 18, 12, 0)  # June 18 2026, noon ET
    with patch("bot.datetime") as mock_dt:
        # Preserve the real datetime class behavior but override .now()
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        mock_dt.now.return_value = now
        mock_dt.fromisoformat = datetime.fromisoformat
        return bot._parse_release_datetime(time_str, date_str)


# ---------------------------------------------------------------------------
# Each supported date format produces a valid ISO-8601 string
# ---------------------------------------------------------------------------

class TestDateFormats:
    """Each supported format parses to a valid ISO-8601 datetime string."""

    def test_date_none_future_time_uses_today(self):
        """When date_str is None and the time is in the future, use today."""
        now = _mock_now(2026, 6, 18, 10, 0)  # 10 AM
        result = _parse_with_frozen_now("6pm", now=now)
        dt = datetime.fromisoformat(result)
        assert dt.month == 6
        assert dt.day == 18
        assert dt.hour == 18

    def test_date_none_past_time_uses_tomorrow(self):
        """When date_str is None and the time has passed, use tomorrow."""
        now = _mock_now(2026, 6, 18, 20, 0)  # 8 PM
        result = _parse_with_frozen_now("6pm", now=now)
        dt = datetime.fromisoformat(result)
        assert dt.month == 6
        assert dt.day == 19
        assert dt.hour == 18

    def test_today(self):
        result = _parse_with_frozen_now("9am", "today")
        dt = datetime.fromisoformat(result)
        assert dt.month == 6
        assert dt.day == 18
        assert dt.hour == 9

    def test_tomorrow(self):
        result = _parse_with_frozen_now("9am", "tomorrow")
        dt = datetime.fromisoformat(result)
        assert dt.month == 6
        assert dt.day == 19
        assert dt.hour == 9

    def test_yyyy_mm_dd(self):
        result = _parse_with_frozen_now("18:00", "2026-07-04")
        dt = datetime.fromisoformat(result)
        assert dt.year == 2026
        assert dt.month == 7
        assert dt.day == 4
        assert dt.hour == 18

    def test_mm_dd(self):
        """MM/DD without year uses current year (or rolls to next if past)."""
        result = _parse_with_frozen_now("6pm", "7/4")
        dt = datetime.fromisoformat(result)
        assert dt.month == 7
        assert dt.day == 4
        assert dt.hour == 18

    def test_mm_dd_yyyy(self):
        result = _parse_with_frozen_now("6pm", "7/4/2027")
        dt = datetime.fromisoformat(result)
        assert dt.year == 2027
        assert dt.month == 7
        assert dt.day == 4

    def test_mm_dd_yy_short(self):
        """Two-digit year gets 2000 added."""
        result = _parse_with_frozen_now("6pm", "7/4/27")
        dt = datetime.fromisoformat(result)
        assert dt.year == 2027
        assert dt.month == 7
        assert dt.day == 4

    def test_month_dd_full(self):
        """'June 25' style."""
        result = _parse_with_frozen_now("6pm", "June 25")
        dt = datetime.fromisoformat(result)
        assert dt.month == 6
        assert dt.day == 25
        assert dt.hour == 18

    def test_month_dd_abbreviated(self):
        """'Jun 25' style."""
        result = _parse_with_frozen_now("6pm", "Jun 25")
        dt = datetime.fromisoformat(result)
        assert dt.month == 6
        assert dt.day == 25

    def test_all_formats_produce_iso8601(self):
        """Every format returns a parseable ISO-8601 string."""
        cases = [
            ("6pm", None),
            ("18:00", "today"),
            ("9am", "tomorrow"),
            ("6pm", "2026-12-25"),
            ("6pm", "12/25"),
            ("6pm", "12/25/2026"),
            ("6pm", "December 25"),
            ("6pm", "Dec 25"),
        ]
        for time_str, date_str in cases:
            result = _parse_with_frozen_now(time_str, date_str)
            # Must not raise
            dt = datetime.fromisoformat(result)
            assert dt.tzinfo is not None, (
                f"format ({time_str!r}, {date_str!r}) must include timezone"
            )


# ---------------------------------------------------------------------------
# Year rollover -- "Month DD" and "MM/DD" roll forward when date is in the past
# ---------------------------------------------------------------------------

class TestYearRollover:
    """Both 'Month DD' and 'MM/DD' roll to next year when the date has passed."""

    def test_month_dd_future_same_year(self):
        """'December 25' on June 18 stays in 2026."""
        now = _mock_now(2026, 6, 18)
        result = _parse_with_frozen_now("6pm", "December 25", now=now)
        dt = datetime.fromisoformat(result)
        assert dt.year == 2026
        assert dt.month == 12
        assert dt.day == 25

    def test_month_dd_past_rolls_to_next_year(self):
        """'January 15' on June 18 rolls to 2027."""
        now = _mock_now(2026, 6, 18)
        result = _parse_with_frozen_now("6pm", "January 15", now=now)
        dt = datetime.fromisoformat(result)
        assert dt.year == 2027
        assert dt.month == 1
        assert dt.day == 15

    def test_mm_dd_future_same_year(self):
        """'12/25' on June 18 stays in 2026."""
        now = _mock_now(2026, 6, 18)
        result = _parse_with_frozen_now("6pm", "12/25", now=now)
        dt = datetime.fromisoformat(result)
        assert dt.year == 2026
        assert dt.month == 12
        assert dt.day == 25

    def test_mm_dd_past_rolls_to_next_year(self):
        """'1/15' on June 18 rolls to 2027."""
        now = _mock_now(2026, 6, 18)
        result = _parse_with_frozen_now("6pm", "1/15", now=now)
        dt = datetime.fromisoformat(result)
        assert dt.year == 2027
        assert dt.month == 1
        assert dt.day == 15

    def test_mm_dd_with_explicit_year_does_not_roll(self):
        """'1/15/2026' does NOT roll even though it's past -- user chose the year."""
        now = _mock_now(2026, 6, 18)
        result = _parse_with_frozen_now("6pm", "1/15/2026", now=now)
        dt = datetime.fromisoformat(result)
        assert dt.year == 2026
        assert dt.month == 1
        assert dt.day == 15

    def test_yyyy_mm_dd_does_not_roll(self):
        """YYYY-MM-DD is explicit; no rollover regardless of past date."""
        now = _mock_now(2026, 6, 18)
        result = _parse_with_frozen_now("6pm", "2025-01-01", now=now)
        dt = datetime.fromisoformat(result)
        assert dt.year == 2025

    def test_month_dd_today_does_not_roll(self):
        """'June 18' on June 18 should stay in 2026 (same day is not past)."""
        now = _mock_now(2026, 6, 18, 8, 0)  # morning
        result = _parse_with_frozen_now("6pm", "June 18", now=now)
        dt = datetime.fromisoformat(result)
        assert dt.year == 2026
        assert dt.month == 6
        assert dt.day == 18

    def test_mm_dd_today_does_not_roll(self):
        """'6/18' on June 18 should stay in 2026."""
        now = _mock_now(2026, 6, 18, 8, 0)
        result = _parse_with_frozen_now("6pm", "6/18", now=now)
        dt = datetime.fromisoformat(result)
        assert dt.year == 2026
        assert dt.month == 6
        assert dt.day == 18


# ---------------------------------------------------------------------------
# Invalid input raises ValueError
# ---------------------------------------------------------------------------

class TestInvalidInput:
    """Invalid time or date strings must raise ValueError."""

    def test_invalid_time_raises(self):
        with pytest.raises(ValueError, match="Cannot parse time"):
            _parse_with_frozen_now("not-a-time", "today")

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            _parse_with_frozen_now("6pm", "not-a-date")

    def test_empty_time_raises(self):
        with pytest.raises(ValueError, match="Cannot parse time"):
            _parse_with_frozen_now("", "today")

    def test_empty_date_raises(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            _parse_with_frozen_now("6pm", "")

    def test_partial_date_raises(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            _parse_with_frozen_now("6pm", "2026-06")


# ---------------------------------------------------------------------------
# Time parsing edge cases
# ---------------------------------------------------------------------------

class TestTimeParsing:
    """Verify various time-string formats are handled correctly."""

    def test_24h_format(self):
        result = _parse_with_frozen_now("18:00", "today")
        dt = datetime.fromisoformat(result)
        assert dt.hour == 18
        assert dt.minute == 0

    def test_12h_pm(self):
        result = _parse_with_frozen_now("6pm", "today")
        dt = datetime.fromisoformat(result)
        assert dt.hour == 18

    def test_12h_am(self):
        result = _parse_with_frozen_now("6am", "today")
        dt = datetime.fromisoformat(result)
        assert dt.hour == 6

    def test_12pm_is_noon(self):
        result = _parse_with_frozen_now("12pm", "today")
        dt = datetime.fromisoformat(result)
        assert dt.hour == 12

    def test_12am_is_midnight(self):
        result = _parse_with_frozen_now("12am", "today")
        dt = datetime.fromisoformat(result)
        assert dt.hour == 0

    def test_time_with_minutes(self):
        result = _parse_with_frozen_now("6:30pm", "today")
        dt = datetime.fromisoformat(result)
        assert dt.hour == 18
        assert dt.minute == 30
