"""Smoke tests for matplotlib/PIL render helpers.

These guard against API drift: they confirm each renderer runs without raising
and produces a Figure or a non-empty PNG, not the visual correctness of output.
"""
import numpy as np
from matplotlib.figure import Figure

from analyzer import (
    render_palette_chart,
    render_hue_saturation_chart,
    render_chart_to_bytesio,
    render_compare_chart,
    render_harmony_chart,
    render_color_info_swatch,
)
from conftest import make_multicolor_image


COLORS = np.array([[200, 40, 40], [40, 200, 40], [40, 40, 200]])
COUNTS = np.array([500, 300, 200])

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_render_palette_chart_returns_figure():
    fig = render_palette_chart(COLORS, COUNTS)
    assert isinstance(fig, Figure)
    render_chart_to_bytesio(fig)  # closes the figure


def test_render_hue_saturation_chart_returns_figure():
    img = make_multicolor_image([(200, 0, 0), (0, 200, 0), (0, 0, 200)], size=(32, 32))
    fig = render_hue_saturation_chart(img)
    assert isinstance(fig, Figure)
    render_chart_to_bytesio(fig)


def test_render_chart_to_bytesio_produces_png():
    fig = render_palette_chart(COLORS, COUNTS)
    buf = render_chart_to_bytesio(fig)
    assert buf.getvalue().startswith(PNG_MAGIC)


def test_render_compare_chart_produces_png():
    buf = render_compare_chart(COLORS, COUNTS, "A", COLORS, COUNTS, "B")
    assert buf.getvalue().startswith(PNG_MAGIC)


def test_render_harmony_chart_produces_png():
    orig = [(200, 40, 40), (40, 200, 40)]
    suggested = [((0, 0, 255), "Complement"), ((255, 255, 0), "Triadic A")]
    buf = render_harmony_chart(orig, suggested)
    assert buf.getvalue().startswith(PNG_MAGIC)


def test_render_color_info_swatch_produces_png():
    harmonies = [((0, 0, 255), "Complement"), ((255, 255, 0), "Triadic")]
    buf = render_color_info_swatch((200, 40, 40), harmonies)
    assert buf.getvalue().startswith(PNG_MAGIC)
