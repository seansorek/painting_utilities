"""Tests for hex parsing and color-space conversion utilities."""
import pytest

from analyzer import (
    parse_hex_color,
    parse_multi_hex_gradient,
    reverse_gradient,
    rgb_to_cmyk,
    nearest_color_name,
)


class TestParseHexColor:
    @pytest.mark.parametrize("value, expected", [
        ("#FF0000", (255, 0, 0)),
        ("#00FF00", (0, 255, 0)),
        ("#0000FF", (0, 0, 255)),
        ("FF0000", (255, 0, 0)),           # no leading '#'
        ("#fff", (255, 255, 255)),         # short form expands
        ("#abc", (0xAA, 0xBB, 0xCC)),      # short form per-channel doubling
        ("  #ffffff  ", (255, 255, 255)),  # surrounding whitespace
        ("#FfAa00", (255, 170, 0)),        # mixed case
    ])
    def test_valid(self, value, expected):
        assert parse_hex_color(value) == expected

    @pytest.mark.parametrize("value", ["#FF", "#FFFF", "#1234567", "", "#"])
    def test_invalid_length_raises(self, value):
        with pytest.raises(ValueError):
            parse_hex_color(value)

    @pytest.mark.parametrize("value", ["#GGGGGG", "#12zz45", "#xyzxyz"])
    def test_non_hex_raises(self, value):
        with pytest.raises(ValueError):
            parse_hex_color(value)


class TestParseMultiHexGradient:
    def test_two_colors_even_spacing(self):
        stops = parse_multi_hex_gradient("#000000, #FFFFFF")
        assert stops == [(0.0, 0, 0, 0), (1.0, 255, 255, 255)]

    def test_three_colors_even_spacing(self):
        stops = parse_multi_hex_gradient("#FF0000, #00FF00, #0000FF")
        positions = [s[0] for s in stops]
        assert positions == [0.0, 0.5, 1.0]
        assert stops[0][1:] == (255, 0, 0)
        assert stops[2][1:] == (0, 0, 255)

    def test_ignores_empty_segments(self):
        stops = parse_multi_hex_gradient("#000, , #fff,")
        assert len(stops) == 2

    @pytest.mark.parametrize("value", ["#FF0000", "", "   ", "#fff,"])
    def test_fewer_than_two_raises(self, value):
        with pytest.raises(ValueError):
            parse_multi_hex_gradient(value)


class TestReverseGradient:
    def test_positions_flipped(self):
        stops = [(0.0, 0, 0, 0), (0.25, 10, 20, 30), (1.0, 255, 255, 255)]
        rev = reverse_gradient(stops)
        assert rev == [(1.0, 0, 0, 0), (0.75, 10, 20, 30), (0.0, 255, 255, 255)]

    def test_colors_unchanged(self):
        stops = [(0.0, 1, 2, 3), (1.0, 4, 5, 6)]
        rev = reverse_gradient(stops)
        assert [s[1:] for s in rev] == [s[1:] for s in stops]


class TestRgbToCmyk:
    @pytest.mark.parametrize("rgb, expected", [
        ((0, 0, 0), (0, 0, 0, 100)),       # black -> all key
        ((255, 255, 255), (0, 0, 0, 0)),   # white -> nothing
        ((255, 0, 0), (0, 100, 100, 0)),   # pure red
        ((0, 255, 0), (100, 0, 100, 0)),   # pure green
        ((0, 0, 255), (100, 100, 0, 0)),   # pure blue
    ])
    def test_conversions(self, rgb, expected):
        assert rgb_to_cmyk(*rgb) == expected

    def test_all_channels_in_range(self):
        for c in rgb_to_cmyk(123, 45, 67):
            assert 0 <= c <= 100


class TestNearestColorName:
    @pytest.mark.parametrize("rgb, name", [
        ((255, 0, 0), "Red"),
        ((0, 255, 0), "Lime"),
        ((0, 0, 255), "Blue"),
        ((0, 0, 0), "Black"),
        ((255, 255, 255), "White"),
    ])
    def test_exact_named_colors(self, rgb, name):
        assert nearest_color_name(rgb) == name

    def test_near_match(self):
        # Very close to pure red should still resolve to Red.
        assert nearest_color_name((254, 1, 2)) == "Red"

    def test_returns_string(self):
        assert isinstance(nearest_color_name((137, 90, 21)), str)
