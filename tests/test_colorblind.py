"""Tests for color blindness simulation and the comparison render."""
import numpy as np
import pytest
from PIL import Image

from analyzer import (
    simulate_colorblindness,
    render_colorblind_comparison,
    _CB_SIM,
)
from conftest import make_solid_image, make_multicolor_image


class TestSimulateColorblindness:
    @pytest.mark.parametrize("cb_type", list(_CB_SIM.keys()))
    def test_preserves_shape_and_mode(self, cb_type):
        img = make_multicolor_image([(200, 0, 0), (0, 200, 0), (0, 0, 200)],
                                    size=(16, 16))
        out = simulate_colorblindness(img, cb_type)
        assert out.size == img.size
        assert out.mode == "RGB"

    @pytest.mark.parametrize("cb_type", list(_CB_SIM.keys()))
    def test_values_in_range(self, cb_type):
        img = make_multicolor_image([(255, 0, 0), (0, 255, 0)], size=(16, 16))
        out = simulate_colorblindness(img, cb_type)
        arr = np.array(out)
        assert arr.dtype == np.uint8
        assert arr.min() >= 0 and arr.max() <= 255

    def test_grayscale_roughly_unchanged(self):
        # Achromatic input has no red/green/blue distinction to lose.
        img = make_solid_image((128, 128, 128), size=(16, 16))
        out = simulate_colorblindness(img, "deuteranopia")
        arr_in = np.array(img).astype(int)
        arr_out = np.array(out).astype(int)
        assert np.abs(arr_in - arr_out).mean() < 10

    def test_unknown_type_raises(self):
        img = make_solid_image(size=(8, 8))
        with pytest.raises(KeyError):
            simulate_colorblindness(img, "not_a_type")


class TestRenderColorblindComparison:
    def test_returns_nonempty_png(self):
        img = make_multicolor_image([(200, 0, 0), (0, 200, 0), (0, 0, 200)],
                                    size=(32, 32))
        buf = render_colorblind_comparison(img)
        data = buf.getvalue()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 100
