"""Tests for palette and gradient export encoders."""
import json
import struct

import pytest

from analyzer import (
    _sanitize_export_name,
    export_ase,
    export_swatches,
    export_gpl,
    export_aco,
    export_css,
    export_tailwind,
    export_gradient_ggr,
    export_gradient_json,
)


COLORS = [(200, 40, 40), (40, 200, 40), (40, 40, 200)]
STOPS = [(0.0, 0, 0, 0), (0.5, 128, 64, 200), (1.0, 255, 255, 255)]


class TestExportAse:
    def test_magic_header(self):
        data = export_ase(COLORS)
        assert data[:4] == b"ASEF"

    def test_version_and_block_count(self):
        data = export_ase(COLORS)
        major, minor, n_blocks = struct.unpack(">HHI", data[4:12])
        assert (major, minor) == (1, 0)
        assert n_blocks == len(COLORS)

    def test_returns_bytes(self):
        assert isinstance(export_ase(COLORS), bytes)


class TestExportSwatches:
    def test_parses_as_json(self):
        data = json.loads(export_swatches(COLORS, name="MyPalette"))
        assert data["name"] == "MyPalette"

    def test_padded_to_thirty(self):
        data = json.loads(export_swatches(COLORS))
        assert len(data["swatches"]) == 30

    def test_color_entries_have_hsv(self):
        data = json.loads(export_swatches(COLORS))
        first = data["swatches"][0]
        for key in ("hue", "saturation", "brightness", "alpha", "colorSpace"):
            assert key in first

    def test_padding_entries_are_null(self):
        data = json.loads(export_swatches(COLORS))
        assert data["swatches"][len(COLORS)] is None


class TestExportGpl:
    def test_header(self):
        text = export_gpl(COLORS, name="Test").decode("utf-8")
        assert text.startswith("GIMP Palette")
        assert "Name: Test" in text

    def test_one_line_per_color(self):
        text = export_gpl(COLORS).decode("utf-8")
        # Header is 4 lines (GIMP Palette, Name, Columns, #) then one per color.
        body = text.splitlines()[4:]
        assert len(body) == len(COLORS)

    def test_rgb_values_present(self):
        text = export_gpl([(255, 0, 0)]).decode("utf-8")
        assert "255" in text and "Red" in text


class TestExportAco:
    def test_starts_with_version1(self):
        data = export_aco(COLORS)
        version, count = struct.unpack(">HH", data[:4])
        assert version == 1
        assert count == len(COLORS)

    def test_contains_version2_block(self):
        data = export_aco(COLORS)
        # Version 2 marker appears after the v1 block.
        v1_size = 4 + len(COLORS) * 10
        version2, count2 = struct.unpack(">HH", data[v1_size:v1_size + 4])
        assert version2 == 2
        assert count2 == len(COLORS)

    def test_returns_bytes(self):
        assert isinstance(export_aco(COLORS), bytes)


class TestExportCss:
    def test_root_block(self):
        text = export_css(COLORS, name="My Palette").decode("utf-8")
        assert text.startswith(":root {")
        assert text.rstrip().endswith("}")

    def test_prefix_slugified(self):
        text = export_css(COLORS, name="My Palette").decode("utf-8")
        assert "--my-palette-1:" in text

    def test_hex_and_rgb_vars(self):
        text = export_css([(255, 0, 0)], name="p").decode("utf-8")
        assert "--p-1: #FF0000;" in text
        assert "--p-1-rgb: 255, 0, 0;" in text


class TestExportTailwind:
    def test_parses_as_json(self):
        data = json.loads(export_tailwind(COLORS, name="brand"))
        colors = data["theme"]["extend"]["colors"]
        assert len(colors) == len(COLORS)

    def test_hex_values(self):
        data = json.loads(export_tailwind([(255, 0, 0)], name="brand"))
        assert data["theme"]["extend"]["colors"]["brand-1"] == "#FF0000"


class TestExportGradientGgr:
    def test_header(self):
        text = export_gradient_ggr(STOPS, name="grad").decode("utf-8")
        lines = text.splitlines()
        assert lines[0] == "GIMP Gradient"
        assert lines[1] == "Name: grad"

    def test_segment_count(self):
        text = export_gradient_ggr(STOPS).decode("utf-8")
        lines = text.splitlines()
        assert int(lines[2]) == len(STOPS) - 1
        # One data line per segment.
        assert len(lines) == 3 + (len(STOPS) - 1)

    def test_sorts_unordered_stops(self):
        unordered = [(1.0, 255, 255, 255), (0.0, 0, 0, 0)]
        text = export_gradient_ggr(unordered).decode("utf-8")
        first_data = text.splitlines()[3]
        assert first_data.startswith("0.000000")


class TestExportGradientJson:
    def test_parses_as_json(self):
        data = json.loads(export_gradient_json(STOPS, name="grad"))
        assert data["name"] == "grad"
        assert len(data["stops"]) == len(STOPS)

    def test_stop_structure(self):
        data = json.loads(export_gradient_json(STOPS))
        first = data["stops"][0]
        for key in ("position", "r", "g", "b"):
            assert key in first

    def test_stops_sorted(self):
        unordered = [(1.0, 255, 255, 255), (0.0, 0, 0, 0)]
        data = json.loads(export_gradient_json(unordered))
        positions = [s["position"] for s in data["stops"]]
        assert positions == sorted(positions)


@pytest.mark.parametrize("fn", [
    export_ase, export_swatches, export_gpl, export_aco, export_css, export_tailwind,
])
def test_palette_exporters_return_bytes(fn):
    assert isinstance(fn(COLORS), bytes)


@pytest.mark.parametrize("fn", [export_gradient_ggr, export_gradient_json])
def test_gradient_exporters_return_bytes(fn):
    assert isinstance(fn(STOPS), bytes)


# ---------------------------------------------------------------------------
# Sanitisation tests (issue #48)
# ---------------------------------------------------------------------------


class TestSanitizeExportName:
    """Unit tests for _sanitize_export_name."""

    def test_strips_newlines(self):
        assert _sanitize_export_name("hello\nworld") == "helloworld"

    def test_strips_carriage_returns(self):
        assert _sanitize_export_name("hello\r\nworld") == "helloworld"

    def test_css_mode_lowercases_and_slugifies(self):
        assert _sanitize_export_name("My Palette", css=True) == "my-palette"

    def test_css_mode_strips_special_chars(self):
        assert _sanitize_export_name("a}; x:y{z", css=True) == "a-xyz"

    def test_css_mode_strips_newlines(self):
        assert _sanitize_export_name("a\nb", css=True) == "ab"

    def test_plain_mode_preserves_spaces(self):
        assert _sanitize_export_name("My Palette") == "My Palette"

    def test_empty_string(self):
        assert _sanitize_export_name("", css=True) == ""


class TestExportCssSanitisation:
    """CSS injection is blocked by _sanitize_export_name(css=True)."""

    def test_injection_stripped(self):
        malicious = "x}; body{background:red"
        text = export_css(COLORS, name=malicious).decode("utf-8")
        assert text.startswith(":root {")
        assert text.count("{") == 1
        assert text.count("}") == 1

    def test_special_chars_removed(self):
        text = export_css(COLORS, name="a;b:c{d}e").decode("utf-8")
        for line in text.splitlines():
            if line.startswith("  --"):
                prop_name = line.split(":")[0].strip().lstrip("-")
                assert ";" not in prop_name
                assert "{" not in prop_name
                assert "}" not in prop_name


class TestExportGplSanitisation:
    """Newline injection in GPL Name header is blocked."""

    def test_newline_in_name_stripped(self):
        text = export_gpl(COLORS, name="Pal\nColumns: 99").decode("utf-8")
        lines = text.splitlines()
        assert lines[1] == "Name: PalColumns: 99"
        assert sum(1 for l in lines if l.startswith("Columns:")) == 1


class TestExportGgrSanitisation:
    """Newline injection in GGR Name header is blocked."""

    def test_newline_in_name_stripped(self):
        text = export_gradient_ggr(STOPS, name="grad\nEvil: header").decode("utf-8")
        lines = text.splitlines()
        assert lines[1] == "Name: gradEvil: header"
        assert len(lines) == 3 + (len(STOPS) - 1)


class TestExportTailwindSanitisation:
    """Tailwind keys are sanitised via CSS mode."""

    def test_special_chars_stripped(self):
        data = json.loads(export_tailwind(COLORS, name="a;b"))
        colors = data["theme"]["extend"]["colors"]
        for key in colors:
            assert ";" not in key
