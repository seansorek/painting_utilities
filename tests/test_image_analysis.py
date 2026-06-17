"""Tests for image loading, color extraction, stats, and adjustment."""
import io

import numpy as np
import pytest
from PIL import Image

import analyzer
from analyzer import (
    MAX_WORKING_DIM,
    load_image_from_bytes,
    extract_dominant_colors,
    compute_stats,
    adjust_image,
)
from conftest import make_solid_image, make_multicolor_image


class TestLoadImageFromBytes:
    def test_roundtrip_png(self, png_bytes):
        img = make_solid_image((10, 20, 30), size=(8, 8))
        loaded = load_image_from_bytes(png_bytes(img))
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
        assert loaded.size == (8, 8)
        assert loaded.getpixel((0, 0)) == (10, 20, 30)

    def test_converts_rgba_to_rgb(self, png_bytes):
        rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
        import io
        buf = io.BytesIO()
        rgba.save(buf, format="PNG")
        loaded = load_image_from_bytes(buf.getvalue())
        assert loaded.mode == "RGB"

    def test_invalid_bytes_raises(self):
        with pytest.raises(Exception):
            load_image_from_bytes(b"not an image")

    def test_small_image_not_resized(self, png_bytes):
        img = make_solid_image((10, 20, 30), size=(64, 48))
        loaded = load_image_from_bytes(png_bytes(img))
        assert loaded.size == (64, 48)

    def test_oversized_image_downscaled_to_working_dim(self, png_bytes):
        # Longest edge exceeds MAX_WORKING_DIM but pixel count stays tiny.
        wide = Image.new("RGB", (MAX_WORKING_DIM + 1000, 10), (50, 100, 150))
        loaded = load_image_from_bytes(png_bytes(wide))
        assert max(loaded.size) == MAX_WORKING_DIM
        # Aspect ratio preserved (very wide image stays very wide).
        assert loaded.width > loaded.height

    def test_decompression_bomb_rejected(self, png_bytes, monkeypatch):
        # Lower the pixel cap so a small test image trips the guard, proving an
        # over-large image is rejected before its raster is fully processed.
        monkeypatch.setattr(analyzer, "MAX_IMAGE_PIXELS", 100)
        bomb = make_solid_image((0, 0, 0), size=(64, 64))  # 4096 px > 100
        with pytest.raises(ValueError, match="too large"):
            load_image_from_bytes(png_bytes(bomb))


class TestExtractDominantColors:
    def test_returns_n_colors(self):
        img = make_multicolor_image([(200, 0, 0), (0, 200, 0), (0, 0, 200)])
        colors, counts = extract_dominant_colors(img, n=3)
        assert colors.shape == (3, 3)
        assert counts.shape == (3,)

    def test_counts_sorted_descending(self):
        img = make_multicolor_image([(200, 0, 0), (0, 200, 0), (0, 0, 200)])
        _, counts = extract_dominant_colors(img, n=4)
        assert list(counts) == sorted(counts, reverse=True)

    def test_counts_sum_to_sample_pixels(self):
        img = make_multicolor_image([(200, 0, 0), (0, 200, 0)])
        _, counts = extract_dominant_colors(img, n=3)
        # KMeans runs on a 200x200 downsample.
        assert counts.sum() == 200 * 200

    def test_dominant_color_of_solid_image(self):
        img = make_solid_image((180, 60, 90))
        colors, counts = extract_dominant_colors(img, n=2)
        top = colors[0]
        assert np.allclose(top, [180, 60, 90], atol=3)

    def test_deterministic(self):
        img = make_multicolor_image([(200, 0, 0), (0, 200, 0), (0, 0, 200)])
        a = extract_dominant_colors(img, n=3)
        b = extract_dominant_colors(img, n=3)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])


class TestComputeStats:
    def test_keys_present(self):
        stats = compute_stats(make_solid_image(size=(20, 30)))
        for key in ("width", "height", "brightness", "contrast",
                    "mean_saturation_pct", "dominant_hue_range"):
            assert key in stats

    def test_dimensions(self):
        stats = compute_stats(make_solid_image(size=(20, 30)))
        assert stats["width"] == 20
        assert stats["height"] == 30

    def test_white_image_bright_and_flat(self):
        stats = compute_stats(make_solid_image((255, 255, 255)))
        assert stats["brightness"] == pytest.approx(255.0, abs=0.5)
        assert stats["contrast"] == pytest.approx(0.0, abs=0.5)
        assert stats["mean_saturation_pct"] == pytest.approx(0.0, abs=0.5)

    def test_black_image_dark(self):
        stats = compute_stats(make_solid_image((0, 0, 0)))
        assert stats["brightness"] == pytest.approx(0.0, abs=0.5)

    def test_hue_range_tuple(self):
        stats = compute_stats(make_solid_image((200, 0, 0)))
        lo, hi = stats["dominant_hue_range"]
        assert 0 <= lo < 360
        assert 0 <= hi < 360


class TestAdjustImage:
    def test_no_change_returns_same_size(self):
        img = make_solid_image(size=(16, 16))
        out = adjust_image(img)
        assert out.size == img.size

    def test_brightness_boost_brightens(self):
        img = make_solid_image((100, 100, 100))
        out = adjust_image(img, brightness_boost=0.5)
        assert out.getpixel((0, 0))[0] > 100

    def test_brightness_reduction_darkens(self):
        img = make_solid_image((100, 100, 100))
        out = adjust_image(img, brightness_boost=-0.5)
        assert out.getpixel((0, 0))[0] < 100

    def test_saturation_boost_returns_image(self):
        img = make_solid_image((200, 50, 50))
        out = adjust_image(img, saturation_boost=0.5)
        assert isinstance(out, Image.Image)
        assert out.size == img.size
