"""Shared fixtures for the painting_utilities test suite.

Test images are kept tiny (KMeans internally resizes to 200x200, so image
content matters more than size) for fast, deterministic runs.
"""
import io

import numpy as np
import pytest
from PIL import Image


def make_solid_image(color=(120, 60, 200), size=(32, 32)) -> Image.Image:
    """An RGB image filled with a single color."""
    return Image.new("RGB", size, color)


def make_gradient_image(size=(64, 64)) -> Image.Image:
    """A horizontal black->white gradient (luminance ramp)."""
    w, h = size
    ramp = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.tile(ramp, (h, 1))
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def make_multicolor_image(colors, size=(64, 64)) -> Image.Image:
    """An image split into equal vertical bands of the given colors."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    band = max(1, w // len(colors))
    for i, c in enumerate(colors):
        arr[:, i * band:(i + 1) * band] = c
    arr[:, len(colors) * band:] = colors[-1]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def solid_image():
    return make_solid_image


@pytest.fixture
def gradient_image():
    return make_gradient_image()


@pytest.fixture
def multicolor_image():
    return make_multicolor_image


@pytest.fixture
def png_bytes():
    """Encode a PIL image to PNG bytes."""
    def _encode(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    return _encode


@pytest.fixture
def sample_palette():
    """A small (colors, counts) pair mimicking extract_dominant_colors output."""
    colors = np.array([
        [200, 40, 40],
        [40, 200, 40],
        [40, 40, 200],
    ], dtype=int)
    counts = np.array([500, 300, 200])
    return colors, counts


@pytest.fixture
def sample_color_list():
    """A plain list of RGB tuples for export functions."""
    return [(200, 40, 40), (40, 200, 40), (40, 40, 200)]


@pytest.fixture
def sample_gradient_stops():
    """Evenly spaced gradient stops from black to white."""
    return [(0.0, 0, 0, 0), (0.5, 128, 64, 200), (1.0, 255, 255, 255)]
