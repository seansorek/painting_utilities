"""Tests for palette-transfer recoloring."""
import numpy as np
from PIL import Image

from analyzer import recolor_image
from conftest import make_multicolor_image, make_solid_image


class TestRecolorImage:
    def test_every_pixel_in_source_palette(self):
        img = make_multicolor_image([(200, 10, 10), (10, 200, 10)], size=(32, 32))
        palette = [(0, 0, 0), (255, 255, 255)]
        out = recolor_image(img, palette)
        pixels = set(map(tuple, np.array(out).reshape(-1, 3)))
        assert pixels <= set(palette)

    def test_size_preserved_when_small(self):
        img = make_solid_image((100, 100, 100), size=(40, 30))
        out = recolor_image(img, [(0, 0, 0), (255, 255, 255)])
        assert out.size == (40, 30)

    def test_large_image_downscaled(self):
        img = make_solid_image((100, 100, 100), size=(2000, 1000))
        out = recolor_image(img, [(0, 0, 0), (255, 255, 255)])
        assert max(out.size) <= 900

    def test_nearest_color_chosen(self):
        # A mid-gray image with a black/white palette should map to white
        # (gray 130 is closer to 255? no -> closer to 0). Use 200 -> white.
        img = make_solid_image((200, 200, 200), size=(8, 8))
        out = recolor_image(img, [(0, 0, 0), (255, 255, 255)])
        assert out.getpixel((0, 0)) == (255, 255, 255)

    def test_output_is_rgb(self):
        img = make_solid_image(size=(8, 8))
        out = recolor_image(img, [(0, 0, 0)])
        assert out.mode == "RGB"
