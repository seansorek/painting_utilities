"""Tests for palette classification, harmony suggestion, and gradient sorting."""
import colorsys

import numpy as np
import pytest

from analyzer import (
    classify_palette_type,
    suggest_harmony_colors,
    palette_to_gradient_stops,
)


def _counts(n):
    return np.array([100] * n)


class TestClassifyPaletteType:
    def test_grayscale_is_monochromatic(self):
        colors = np.array([[10, 10, 10], [128, 128, 128], [240, 240, 240]])
        assert classify_palette_type(colors, _counts(3)) == "Monochromatic"

    def test_single_saturated_hue_is_monochromatic(self):
        # Only one saturated color -> fewer than 2 saturated hues.
        colors = np.array([[200, 0, 0], [20, 20, 20], [240, 240, 240]])
        assert classify_palette_type(colors, _counts(3)) == "Monochromatic"

    def test_red_and_cyan_is_complementary(self):
        colors = np.array([[200, 0, 0], [0, 200, 200]])
        assert classify_palette_type(colors, _counts(2)) == "Complementary"

    def test_close_hues_is_analogous(self):
        # Red (hue 0) and orange (hue ~45) sit in the 30-60 deg Analogous band.
        red = [200, 0, 0]
        orange = [200, 150, 0]
        colors = np.array([red, orange])
        assert classify_palette_type(colors, _counts(2)) == "Analogous"

    def test_returns_known_type(self):
        colors = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]])
        result = classify_palette_type(colors, _counts(3))
        assert result in {
            "Monochromatic", "Analogous", "Complementary",
            "Split Complementary", "Triadic", "Polychromatic",
        }


class TestSuggestHarmonyColors:
    def test_returns_type_and_list(self):
        colors = np.array([[200, 0, 0], [0, 200, 200]])
        ptype, suggestions = suggest_harmony_colors(colors, _counts(2))
        assert isinstance(ptype, str)
        assert isinstance(suggestions, list)

    def test_grayscale_yields_no_suggestions(self):
        colors = np.array([[10, 10, 10], [128, 128, 128], [240, 240, 240]])
        ptype, suggestions = suggest_harmony_colors(colors, _counts(3))
        assert suggestions == []

    def test_suggestion_shape(self):
        colors = np.array([[200, 0, 0], [50, 50, 50]])
        _, suggestions = suggest_harmony_colors(colors, _counts(2))
        for rgb, label in suggestions:
            assert len(rgb) == 3
            assert all(0 <= c <= 255 for c in rgb)
            assert isinstance(label, str)

    def test_complementary_suggests_triadic(self):
        colors = np.array([[200, 0, 0], [0, 200, 200]])
        ptype, suggestions = suggest_harmony_colors(colors, _counts(2))
        assert ptype == "Complementary"
        labels = [lbl for _, lbl in suggestions]
        assert any("Triadic" in lbl for lbl in labels)


class TestPaletteToGradientStops:
    def test_positions_span_zero_to_one(self):
        colors = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]])
        stops = palette_to_gradient_stops(colors, _counts(3))
        positions = [s[0] for s in stops]
        assert positions[0] == 0.0
        assert positions[-1] == 1.0

    def test_single_color_no_div_by_zero(self):
        colors = np.array([[123, 45, 67]])
        stops = palette_to_gradient_stops(colors, _counts(1))
        assert len(stops) == 1
        assert stops[0] == (0.0, 123, 45, 67)

    def test_sort_by_value_ascending(self):
        colors = np.array([[255, 255, 255], [0, 0, 0], [128, 128, 128]])
        stops = palette_to_gradient_stops(colors, _counts(3), sort_by="value")
        values = []
        for _, r, g, b in stops:
            _, _, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            values.append(v)
        assert values == sorted(values)

    def test_sort_by_luminance_ascending(self):
        colors = np.array([[255, 255, 255], [0, 0, 0], [200, 50, 50]])
        stops = palette_to_gradient_stops(colors, _counts(3), sort_by="luminance")
        lums = [0.299 * r + 0.587 * g + 0.114 * b for _, r, g, b in stops]
        assert lums == sorted(lums)

    @pytest.mark.parametrize("sort_by", ["value", "luminance", "hue", "saturation"])
    def test_all_sort_modes_run(self, sort_by):
        colors = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]])
        stops = palette_to_gradient_stops(colors, _counts(3), sort_by=sort_by)
        assert len(stops) == 3
