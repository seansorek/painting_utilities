"""Smoke tests for matplotlib/PIL render helpers.

These guard against API drift: they confirm each renderer runs without raising
and produces a non-empty PNG, not the visual correctness of output. They also
guard against matplotlib Figure leaks on render failures (see issue #35).
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pytest

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


def test_render_palette_chart_produces_png():
    buf = render_palette_chart(COLORS, COUNTS)
    assert isinstance(buf, io.BytesIO)
    assert buf.getvalue().startswith(PNG_MAGIC)


def test_render_hue_saturation_chart_produces_png():
    img = make_multicolor_image([(200, 0, 0), (0, 200, 0), (0, 0, 200)], size=(32, 32))
    buf = render_hue_saturation_chart(img)
    assert isinstance(buf, io.BytesIO)
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


def test_render_palette_chart_closes_figure_on_savefig_failure(monkeypatch):
    """If fig.savefig raises, the figure must still be closed (issue #35)."""
    from matplotlib.figure import Figure

    def _boom(self, *args, **kwargs):
        raise MemoryError("simulated savefig failure")

    monkeypatch.setattr(Figure, "savefig", _boom)

    fignums_before = len(plt.get_fignums())
    with pytest.raises(MemoryError):
        render_palette_chart(COLORS, COUNTS)
    assert len(plt.get_fignums()) == fignums_before


def test_render_hue_saturation_chart_closes_figure_on_savefig_failure(monkeypatch):
    """If fig.savefig raises, the figure must still be closed (issue #35)."""
    from matplotlib.figure import Figure

    def _boom(self, *args, **kwargs):
        raise MemoryError("simulated savefig failure")

    monkeypatch.setattr(Figure, "savefig", _boom)

    img = make_multicolor_image([(200, 0, 0), (0, 200, 0), (0, 0, 200)], size=(32, 32))
    fignums_before = len(plt.get_fignums())
    with pytest.raises(MemoryError):
        render_hue_saturation_chart(img)
    assert len(plt.get_fignums()) == fignums_before


def test_render_palette_chart_closes_figure_on_construction_failure(monkeypatch):
    """If drawing raises before savefig is even reached, the figure must still close."""
    import analyzer

    def _boom(*args, **kwargs):
        raise ValueError("simulated construction failure")

    monkeypatch.setattr(analyzer, "nearest_color_name", _boom)

    fignums_before = len(plt.get_fignums())
    with pytest.raises(ValueError):
        render_palette_chart(COLORS, COUNTS)
    assert len(plt.get_fignums()) == fignums_before


def test_render_chart_to_bytesio_produces_png():
    """render_chart_to_bytesio remains a low-level helper (no longer closes the figure)."""
    fig, ax = plt.subplots()
    try:
        buf = render_chart_to_bytesio(fig)
        assert buf.getvalue().startswith(PNG_MAGIC)
    finally:
        plt.close(fig)
