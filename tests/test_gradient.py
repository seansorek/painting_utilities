"""Tests for gradient mapping, presets, and gradient preview rendering."""
import numpy as np
import pytest
from PIL import Image

from analyzer import (
    apply_gradient_map,
    GRADIENT_PRESETS,
    render_gradient_preview,
)
from conftest import make_solid_image


class TestApplyGradientMap:
    def test_output_size_matches_input(self):
        img = make_solid_image((128, 128, 128), size=(24, 18))
        stops = [(0.0, 0, 0, 0), (1.0, 255, 255, 255)]
        out = apply_gradient_map(img, stops)
        assert out.size == img.size
        assert out.mode == "RGB"

    def test_black_pixel_maps_to_first_stop(self):
        img = make_solid_image((0, 0, 0), size=(4, 4))
        stops = [(0.0, 10, 20, 30), (1.0, 200, 210, 220)]
        out = apply_gradient_map(img, stops)
        assert out.getpixel((0, 0)) == (10, 20, 30)

    def test_white_pixel_maps_to_last_stop(self):
        img = make_solid_image((255, 255, 255), size=(4, 4))
        stops = [(0.0, 10, 20, 30), (1.0, 200, 210, 220)]
        out = apply_gradient_map(img, stops)
        assert out.getpixel((0, 0)) == (200, 210, 220)

    def test_unsorted_stops_handled(self):
        img = make_solid_image((0, 0, 0), size=(4, 4))
        stops = [(1.0, 200, 210, 220), (0.0, 10, 20, 30)]
        out = apply_gradient_map(img, stops)
        assert out.getpixel((0, 0)) == (10, 20, 30)

    def test_uses_preset(self):
        img = make_solid_image((128, 128, 128), size=(8, 8))
        out = apply_gradient_map(img, GRADIENT_PRESETS["fire"])
        assert isinstance(out, Image.Image)


class TestGradientPresets:
    @pytest.mark.parametrize("name", list(GRADIENT_PRESETS.keys()))
    def test_preset_starts_at_zero_ends_at_one(self, name):
        stops = GRADIENT_PRESETS[name]
        positions = [s[0] for s in stops]
        assert positions[0] == 0.0
        assert positions[-1] == 1.0

    @pytest.mark.parametrize("name", list(GRADIENT_PRESETS.keys()))
    def test_preset_positions_sorted(self, name):
        positions = [s[0] for s in GRADIENT_PRESETS[name]]
        assert positions == sorted(positions)

    @pytest.mark.parametrize("name", list(GRADIENT_PRESETS.keys()))
    def test_preset_rgb_in_range(self, name):
        for _, r, g, b in GRADIENT_PRESETS[name]:
            assert all(0 <= c <= 255 for c in (r, g, b))

    def test_expected_presets_present(self):
        assert {"fire", "ocean", "forest", "grayscale"} <= set(GRADIENT_PRESETS)


class TestRenderGradientPreview:
    def test_returns_nonempty_png(self):
        buf = render_gradient_preview([(0.0, 0, 0, 0), (1.0, 255, 255, 255)])
        data = buf.getvalue()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 0

    def test_custom_dimensions(self):
        buf = render_gradient_preview(
            [(0.0, 0, 0, 0), (1.0, 255, 255, 255)], width=100, height=20
        )
        img = Image.open(buf)
        assert img.size == (100, 20)
