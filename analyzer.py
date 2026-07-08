import io
import re
import json
import math
import re
import struct
import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Image safety policy (single source of truth)
#
# All untrusted images enter through load_image_from_bytes(), which is the one
# place that enforces these limits. Every downstream function (compute_stats,
# apply_gradient_map, KMeans sampling, rendering, ...) therefore only ever sees
# an already-bounded image, so none of them can be used as a decompression-bomb
# vector. Purpose-specific resizes elsewhere (KMeans 200x200 sample, colorblind
# thumbnail, recolor output cap) are about quality/output size, not safety.
# ---------------------------------------------------------------------------

# Reject images that decode to more than this many pixels (a tiny compressed
# file can otherwise expand to gigabytes of raster and OOM the host).
MAX_IMAGE_PIXELS = 40_000_000  # ~40 MP

# Also raise PIL's own guard so a malformed/oversized raster is refused even if
# the declared dimensions slipped past the explicit check below.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Longest edge every loaded image is downscaled to before any processing.
MAX_WORKING_DIM = 2048

# ---------------------------------------------------------------------------
# Named color reference table (~140 entries for precise nearest-color naming)
# ---------------------------------------------------------------------------
_NAMED_COLORS = [
    # Reds
    ("Crimson",           220,  20,  60),
    ("Red",               255,   0,   0),
    ("Dark Red",          139,   0,   0),
    ("Firebrick",         178,  34,  34),
    ("Indian Red",        205,  92,  92),
    ("Salmon",            250, 128, 114),
    ("Light Salmon",      255, 160, 122),
    ("Tomato",            255,  99,  71),
    ("Vermilion",         227,  66,  52),
    ("Scarlet",           255,  36,   0),
    ("Maroon",            128,   0,   0),
    # Oranges
    ("Coral",             255, 127,  80),
    ("Coral Pink",        248, 131, 121),
    ("Orange Red",        255,  69,   0),
    ("Burnt Orange",      204,  85,   0),
    ("Dark Orange",       255, 140,   0),
    ("Orange",            255, 165,   0),
    ("Rust",              183,  65,  14),
    ("Terracotta",        226, 114,  78),
    # Yellows
    ("Amber",             255, 191,   0),
    ("Gold",              255, 215,   0),
    ("Yellow",            255, 255,   0),
    ("Light Yellow",      255, 255, 224),
    ("Khaki",             240, 230, 140),
    ("Dark Khaki",        189, 183, 107),
    ("Mustard",           255, 219,  88),
    ("Chartreuse",        127, 255,   0),
    ("Lawn Green",        124, 252,   0),
    # Yellow-Greens
    ("Yellow Green",      154, 205,  50),
    ("Lime Green",         50, 205,  50),
    # Greens
    ("Lime",                0, 255,   0),
    ("Green",               0, 128,   0),
    ("Dark Green",          0, 100,   0),
    ("Forest Green",       34, 139,  34),
    ("Sea Green",          46, 139,  87),
    ("Medium Sea Green",   60, 179, 113),
    ("Spring Green",        0, 255, 127),
    ("Medium Spring Green", 0, 250, 154),
    ("Light Green",       144, 238, 144),
    ("Pale Green",        152, 251, 152),
    ("Mint",              152, 255, 152),
    ("Sage",              188, 184, 138),
    ("Olive",             128, 128,   0),
    ("Dark Olive Green",   85, 107,  47),
    ("Olive Drab",        107, 142,  35),
    # Teals / Aquas
    ("Teal",                0, 128, 128),
    ("Teal Green",          0, 128, 105),
    ("Dark Cyan",           0, 139, 139),
    ("Light Sea Green",    32, 178, 170),
    ("Medium Aquamarine", 102, 205, 170),
    ("Aquamarine",        127, 255, 212),
    ("Turquoise",          64, 224, 208),
    ("Medium Turquoise",   72, 209, 204),
    ("Dark Turquoise",      0, 206, 209),
    ("Cadet Blue",         95, 158, 160),
    # Cyans
    ("Cyan",                0, 255, 255),
    ("Light Cyan",        224, 255, 255),
    ("Pale Turquoise",    175, 238, 238),
    # Blues
    ("Steel Blue",         70, 130, 180),
    ("Cornflower Blue",   100, 149, 237),
    ("Dodger Blue",        30, 144, 255),
    ("Deep Sky Blue",       0, 191, 255),
    ("Sky Blue",          135, 206, 235),
    ("Light Sky Blue",    135, 206, 250),
    ("Light Blue",        173, 216, 230),
    ("Powder Blue",       176, 224, 230),
    ("Royal Blue",         65, 105, 225),
    ("Blue",                0,   0, 255),
    ("Medium Blue",         0,   0, 205),
    ("Dark Blue",           0,   0, 139),
    ("Navy",                0,   0, 128),
    ("Electric Blue",       0, 115, 207),
    ("Periwinkle",        204, 204, 255),
    # Slate / Indigo
    ("Slate Blue",        106,  90, 205),
    ("Medium Slate Blue", 123, 104, 238),
    ("Dark Slate Blue",    72,  61, 139),
    ("Indigo",             75,   0, 130),
    ("Blue Violet",       138,  43, 226),
    # Purples / Violets
    ("Violet",            238, 130, 238),
    ("Dark Violet",       148,   0, 211),
    ("Purple",            128,   0, 128),
    ("Dark Orchid",       153,  50, 204),
    ("Medium Orchid",     186,  85, 211),
    ("Orchid",            218, 112, 214),
    ("Medium Purple",     147, 112, 219),
    ("Plum",              221, 160, 221),
    ("Lavender",          230, 230, 250),
    ("Thistle",           216, 191, 216),
    ("Wisteria",          201, 160, 220),
    ("Lilac",             200, 162, 200),
    ("Mauve",             224, 176, 255),
    # Pinks / Magentas
    ("Magenta",           255,   0, 255),
    ("Hot Pink",          255, 105, 180),
    ("Deep Pink",         255,  20, 147),
    ("Pink",              255, 192, 203),
    ("Light Pink",        255, 182, 193),
    ("Pale Violet Red",   219, 112, 147),
    ("Medium Violet Red", 199,  21, 133),
    ("Rose Gold",         183, 110, 121),
    ("Dusty Rose",        220, 174, 150),
    # Browns / Earth
    ("Brown",             165,  42,  42),
    ("Saddle Brown",      139,  69,  19),
    ("Sienna",            160,  82,  45),
    ("Chocolate",         210, 105,  30),
    ("Peru",              205, 133,  63),
    ("Sandy Brown",       244, 164,  96),
    ("Burlywood",         222, 184, 135),
    ("Tan",               210, 180, 140),
    ("Wheat",             245, 222, 179),
    ("Moccasin",          255, 228, 181),
    ("Copper",            184, 115,  51),
    ("Bronze",            205, 127,  50),
    # Whites / Creams
    ("White",             255, 255, 255),
    ("Snow",              255, 250, 250),
    ("Ivory",             255, 255, 240),
    ("Cream",             255, 253, 208),
    ("Beige",             245, 245, 220),
    ("Linen",             250, 240, 230),
    ("Floral White",      255, 250, 240),
    ("Lavender Blush",    255, 240, 245),
    ("Misty Rose",        255, 228, 225),
    ("Antique White",     250, 235, 215),
    # Grays
    ("Gainsboro",         220, 220, 220),
    ("Light Gray",        211, 211, 211),
    ("Silver",            192, 192, 192),
    ("Dark Silver",       169, 169, 169),
    ("Gray",              128, 128, 128),
    ("Dim Gray",          105, 105, 105),
    ("Dark Gray",          64,  64,  64),
    ("Charcoal",           54,  69,  79),
    ("Light Slate Gray",  119, 136, 153),
    ("Slate Gray",        112, 128, 144),
    ("Dark Slate Gray",    47,  79,  79),
    # Black
    ("Black",               0,   0,   0),
]


GRADIENT_PRESETS: dict[str, list[tuple[float, int, int, int]]] = {
    "fire":      [(0.0,   0,   0,   0), (0.33, 180,  30,   0), (0.66, 255, 160,   0), (1.0, 255, 255, 180)],
    "ocean":     [(0.0,   0,   0,  40), (0.4,    0,  60, 130), (0.75,   0, 160, 200), (1.0, 200, 240, 255)],
    "forest":    [(0.0,   5,  20,   5), (0.35,  20,  80,  20), (0.7,   80, 160,  40), (1.0, 200, 230, 140)],
    "amethyst":  [(0.0,  10,   0,  20), (0.4,   80,   0, 150), (0.75, 180,  80, 220), (1.0, 240, 200, 255)],
    "grayscale": [(0.0,   0,   0,   0), (1.0,  255, 255, 255)],
    "sunset":    [(0.0,  20,   0,  40), (0.3,  180,  30,  60), (0.65, 255, 130,  30), (1.0, 255, 220, 120)],
    "ice":       [(0.0,   0,  10,  40), (0.45,  80, 160, 220), (0.8,  190, 230, 255), (1.0, 240, 250, 255)],
}

# ---------------------------------------------------------------------------
# Color blindness simulation matrices (Brettel/Vienot method, LMS space)
# ---------------------------------------------------------------------------
_RGB_TO_LMS = np.array([
    [17.8824,   43.5161,   4.11935],
    [ 3.45565,  27.1554,   3.86714],
    [ 0.0299566, 0.184309, 1.46709],
], dtype=np.float64)

_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)

_CB_SIM: dict[str, np.ndarray] = {
    "protanopia": np.array([
        [0,        2.02344, -2.52581],
        [0,        1,        0      ],
        [0,        0,        1      ],
    ], dtype=np.float64),
    "deuteranopia": np.array([
        [1,        0,        0      ],
        [0.494207, 0,        1.24827],
        [0,        0,        1      ],
    ], dtype=np.float64),
    "tritanopia": np.array([
        [1,         0,        0],
        [0,         1,        0],
        [-0.395913, 0.801109, 0],
    ], dtype=np.float64),
}


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.strip().lstrip("#").upper()
    if len(s) == 3:
        s = s[0] * 2 + s[1] * 2 + s[2] * 2
    if len(s) != 6:
        raise ValueError(f"Invalid hex color '{hex_str}': must be #RGB or #RRGGBB")
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        raise ValueError(f"Invalid hex color '{hex_str}': contains non-hex characters")


def parse_multi_hex_gradient(hex_list_str: str) -> list[tuple[float, int, int, int]]:
    """Parse a comma-separated list of hex colors into evenly-spaced gradient stops."""
    parts = [p.strip() for p in hex_list_str.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("Provide at least 2 hex colors separated by commas.")
    colors = [parse_hex_color(p) for p in parts]
    n = len(colors)
    return [(i / (n - 1), r, g, b) for i, (r, g, b) in enumerate(colors)]


def reverse_gradient(
    stops: list[tuple[float, int, int, int]]
) -> list[tuple[float, int, int, int]]:
    """Flip a gradient so shadows become highlights and vice versa."""
    return [(1.0 - pos, r, g, b) for pos, r, g, b in stops]


def nearest_color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    best, best_dist = _NAMED_COLORS[0][0], float("inf")
    for name, nr, ng, nb in _NAMED_COLORS:
        d = (r - nr) ** 2 + (g - ng) ** 2 + (b - nb) ** 2
        if d < best_dist:
            best_dist, best = d, name
    return best


def rgb_to_cmyk(r: int, g: int, b: int) -> tuple[int, int, int, int]:
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    k = 1 - max(r_, g_, b_)
    if k == 1.0:
        return 0, 0, 0, 100
    c = (1 - r_ - k) / (1 - k)
    m = (1 - g_ - k) / (1 - k)
    y = (1 - b_ - k) / (1 - k)
    return round(c * 100), round(m * 100), round(y * 100), round(k * 100)


def adjust_image(
    img: Image.Image,
    saturation_boost: float = 0.0,
    brightness_boost: float = 0.0,
) -> Image.Image:
    """Adjust saturation and brightness. Values in [-1, 1]: 0 = unchanged."""
    if saturation_boost != 0.0:
        img = ImageEnhance.Color(img).enhance(max(0.0, 1.0 + saturation_boost))
    if brightness_boost != 0.0:
        img = ImageEnhance.Brightness(img).enhance(max(0.0, 1.0 + brightness_boost))
    return img


# ---------------------------------------------------------------------------
# Image loading / extraction / stats
# ---------------------------------------------------------------------------

def _downscale_to_working(img: Image.Image, max_dim: int = MAX_WORKING_DIM) -> Image.Image:
    """Downscale so the longest edge is at most max_dim; leave smaller images as-is."""
    longest = max(img.width, img.height)
    if longest <= max_dim:
        return img
    ratio = max_dim / longest
    return img.resize(
        (max(1, round(img.width * ratio)), max(1, round(img.height * ratio))),
        Image.LANCZOS,
    )


def load_image_from_bytes(data: bytes, max_dim: int = MAX_WORKING_DIM) -> Image.Image:
    """Decode untrusted image bytes into a safe, size-bounded RGB image.

    This is the single choke point for image safety: it rejects oversized images
    (decompression-bomb guard) using the declared dimensions before the full
    raster is materialised, then downscales to a bounded working resolution so
    no downstream operation ever touches an unbounded array.
    """
    img = Image.open(io.BytesIO(data))
    width, height = img.size
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is too large to process: {width}×{height} px exceeds the "
            f"{MAX_IMAGE_PIXELS:,}-pixel limit."
        )
    img = img.convert("RGB")
    return _downscale_to_working(img, max_dim)


def extract_dominant_colors(
    img: Image.Image, n: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    small = img.resize((200, 200), Image.LANCZOS)
    pixels = np.array(small).reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    km.fit(pixels)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(km.labels_, minlength=n)
    mask = counts > 0
    centers, counts = centers[mask], counts[mask]
    order = np.argsort(counts)[::-1]
    return centers[order], counts[order]


def compute_stats(img: Image.Image) -> dict:
    arr = np.array(img).astype(float)
    brightness = arr.mean()
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    contrast = lum.std()

    hsv = np.array(img.convert("HSV"))
    hue_vals = hsv[:, :, 0].flatten()
    sat_vals = hsv[:, :, 1].flatten()
    mean_sat_pct = sat_vals.mean() / 255 * 100

    val_vals = hsv[:, :, 2].flatten()
    mask = (sat_vals / 255 >= 0.15) & (val_vals / 255 >= 0.10)
    filtered_hue = hue_vals[mask]

    if len(filtered_hue) == 0:
        dom_hue_start, dom_hue_end = None, None
    else:
        hue_deg = filtered_hue.astype(float) / 255 * 360
        hist, edges = np.histogram(hue_deg, bins=36, range=(0, 360))
        best_start, best_sum = 0, -1
        for i in range(36):
            window = sum(hist[(i + j) % 36] for j in range(6))
            if window > best_sum:
                best_sum, best_start = window, i
        dom_hue_start = int(edges[best_start])
        dom_hue_end = (dom_hue_start + 60) % 360

    return {
        "width": img.width,
        "height": img.height,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "mean_saturation_pct": round(mean_sat_pct, 1),
        "dominant_hue_range": (dom_hue_start, dom_hue_end),
    }


# ---------------------------------------------------------------------------
# Palette classification & harmony
# ---------------------------------------------------------------------------

def classify_palette_type(colors: np.ndarray, counts: np.ndarray) -> str:
    saturated_hues = []
    for rgb in colors:
        h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        if s >= 0.15 and v >= 0.10:
            saturated_hues.append(h * 360)

    if len(saturated_hues) < 2:
        return "Monochromatic"

    def _circ_dist(a, b):
        d = abs(a - b) % 360
        return d if d <= 180 else 360 - d

    max_spread = max(_circ_dist(a, b) for i, a in enumerate(saturated_hues)
                     for b in saturated_hues[i + 1:])

    if max_spread < 30:
        return "Monochromatic"
    if max_spread < 60:
        return "Analogous"

    for i, a in enumerate(saturated_hues):
        for b in saturated_hues[i + 1:]:
            if abs(_circ_dist(a, b) - 180) <= 30:
                others = [h for h in saturated_hues if h != a and h != b]
                for c in others:
                    if abs(_circ_dist(a, c) - 150) <= 30 or abs(_circ_dist(b, c) - 150) <= 30:
                        return "Split Complementary"
                return "Complementary"

    hues = sorted(saturated_hues)
    for i, a in enumerate(hues):
        for b in hues[i + 1:]:
            if abs(_circ_dist(a, b) - 120) <= 30:
                for c in hues:
                    if c != a and c != b:
                        if abs(_circ_dist(b, c) - 120) <= 30 and abs(_circ_dist(a, c) - 120) <= 30:
                            return "Triadic"

    return "Polychromatic"


def suggest_harmony_colors(
    colors: np.ndarray, counts: np.ndarray
) -> tuple[str, list[tuple[tuple[int, int, int], str]]]:
    """
    Returns (palette_type, [(rgb, label), ...]) — colors that would harmonize
    with the existing palette but are not already present.
    """
    palette_type = classify_palette_type(colors, counts)
    total = counts.sum()

    dominant_h: float | None = None
    dominant_s = 0.0
    dominant_v = 0.7

    for rgb, cnt in zip(colors, counts):
        h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        if s >= 0.15 and v >= 0.10 and s > dominant_s:
            dominant_s = s
            dominant_h = h
            dominant_v = max(0.5, v)

    if dominant_h is None:
        return palette_type, []

    use_s = max(dominant_s, 0.65)
    use_v = max(dominant_v, 0.60)

    def hue_rgb(h: float) -> tuple[int, int, int]:
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, use_s, use_v)
        return int(r * 255), int(g * 255), int(b * 255)

    if palette_type in ("Monochromatic", "Analogous"):
        suggestions = [
            (hue_rgb(dominant_h + 0.5), "Complement"),
            (hue_rgb(dominant_h + 5 / 12), "Split Complement A"),
            (hue_rgb(dominant_h + 7 / 12), "Split Complement B"),
        ]
    elif palette_type == "Complementary":
        suggestions = [
            (hue_rgb(dominant_h + 1 / 3), "Triadic A"),
            (hue_rgb(dominant_h + 2 / 3), "Triadic B"),
        ]
    elif palette_type in ("Split Complementary", "Triadic"):
        suggestions = [
            (hue_rgb(dominant_h + 1 / 12), "Analogous +30°"),
            (hue_rgb(dominant_h - 1 / 12), "Analogous -30°"),
            (hue_rgb(dominant_h + 0.25),   "Tetradic"),
        ]
    else:  # Polychromatic
        r, g, b = colorsys.hsv_to_rgb(dominant_h, 0.15, 0.92)
        light = (int(r * 255), int(g * 255), int(b * 255))
        r2, g2, b2 = colorsys.hsv_to_rgb(dominant_h, 0.90, 0.30)
        deep = (int(r2 * 255), int(g2 * 255), int(b2 * 255))
        suggestions = [(light, "Light Tint"), (deep, "Deep Shade")]

    return palette_type, suggestions


# ---------------------------------------------------------------------------
# Gradient operations
# ---------------------------------------------------------------------------

def apply_gradient_map(
    img: Image.Image,
    gradient_stops: list[tuple[float, int, int, int]],
) -> Image.Image:
    stops = sorted(gradient_stops, key=lambda s: s[0])
    lut_r = np.zeros(256, dtype=np.uint8)
    lut_g = np.zeros(256, dtype=np.uint8)
    lut_b = np.zeros(256, dtype=np.uint8)
    n = len(stops)
    for i in range(256):
        t = i / 255.0
        for k in range(n - 1):
            pos0, r0, g0, b0 = stops[k]
            pos1, r1, g1, b1 = stops[k + 1]
            if t <= pos1 or k == n - 2:
                span = pos1 - pos0
                local_t = 0.0 if span == 0 else max(0.0, min(1.0, (t - pos0) / span))
                lut_r[i] = round(r0 + (r1 - r0) * local_t)
                lut_g[i] = round(g0 + (g1 - g0) * local_t)
                lut_b[i] = round(b0 + (b1 - b0) * local_t)
                break
    arr = np.array(img, dtype=np.float32)
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
    out = np.stack([lut_r[lum], lut_g[lum], lut_b[lum]], axis=-1)
    return Image.fromarray(out, mode="RGB")


def palette_to_gradient_stops(
    colors: np.ndarray, counts: np.ndarray, sort_by: str = "value"
) -> list[tuple[float, int, int, int]]:
    def _key(rgb):
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if sort_by == "luminance":
            return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        if sort_by == "hue":
            return h
        if sort_by == "saturation":
            return s
        return v

    order = np.argsort([_key(c) for c in colors])
    sorted_colors = colors[order]
    n = len(sorted_colors)
    return [
        (i / max(n - 1, 1), int(rgb[0]), int(rgb[1]), int(rgb[2]))
        for i, rgb in enumerate(sorted_colors)
    ]


# ---------------------------------------------------------------------------
# Color blindness simulation
# ---------------------------------------------------------------------------

def simulate_colorblindness(img: Image.Image, cb_type: str) -> Image.Image:
    """
    Simulate how an image looks for a given color blindness type.
    cb_type: 'deuteranopia', 'protanopia', or 'tritanopia'
    """
    sim_matrix = _CB_SIM[cb_type]
    arr = np.array(img, dtype=np.float64) / 255.0

    # sRGB gamma decode
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    h, w, _ = linear.shape
    pixels = linear.reshape(-1, 3)

    # RGB → LMS → simulated LMS → RGB
    lms = pixels @ _RGB_TO_LMS.T
    sim_lms = lms @ sim_matrix.T
    sim_rgb = sim_lms @ _LMS_TO_RGB.T

    # sRGB gamma encode
    sim_rgb = np.clip(sim_rgb, 0.0, 1.0)
    encoded = np.where(
        sim_rgb <= 0.0031308,
        sim_rgb * 12.92,
        1.055 * sim_rgb ** (1.0 / 2.4) - 0.055,
    )
    out = np.clip(encoded * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out.reshape(h, w, 3), mode="RGB")


def render_colorblind_comparison(img: Image.Image) -> io.BytesIO:
    """4-panel grid: original + deuteranopia + protanopia + tritanopia."""
    max_dim = 380
    ratio = min(1.0, max_dim / img.width, max_dim / img.height)
    thumb_w = max(1, int(img.width * ratio))
    thumb_h = max(1, int(img.height * ratio))
    thumb = img.resize((thumb_w, thumb_h), Image.LANCZOS)

    cb_types = ["deuteranopia", "protanopia", "tritanopia"]
    panels = [thumb] + [simulate_colorblindness(thumb, t) for t in cb_types]
    labels = ["Original", "Deuteranopia\n(red-green)", "Protanopia\n(red)", "Tritanopia\n(blue-yellow)"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    fig.patch.set_facecolor("#1a1a1a")
    for ax, panel, label in zip(axes.flatten(), panels, labels):
        ax.imshow(np.array(panel))
        ax.set_title(label, color="white", fontsize=9, pad=5)
        ax.axis("off")
        ax.set_facecolor("#1a1a1a")

    fig.tight_layout(pad=1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#1a1a1a")
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Recolor (palette transfer)
# ---------------------------------------------------------------------------

def recolor_image(
    target: Image.Image, source_colors: list[tuple[int, int, int]]
) -> Image.Image:
    """Replace every pixel in target with the nearest color from source_colors."""
    max_dim = 900
    if max(target.width, target.height) > max_dim:
        ratio = max_dim / max(target.width, target.height)
        target = target.resize(
            (max(1, int(target.width * ratio)), max(1, int(target.height * ratio))),
            Image.LANCZOS,
        )

    arr = np.array(target, dtype=np.int32)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)
    palette = np.array(source_colors, dtype=np.int32)

    chunk = 60_000
    nearest = np.zeros(len(pixels), dtype=np.intp)
    for i in range(0, len(pixels), chunk):
        seg = pixels[i : i + chunk].astype(np.float32)
        pal = palette.astype(np.float32)
        dists = np.sum((seg[:, None, :] - pal[None, :, :]) ** 2, axis=2)
        nearest[i : i + chunk] = np.argmin(dists, axis=1)

    out = palette[nearest].astype(np.uint8)
    return Image.fromarray(out.reshape(h, w, 3), mode="RGB")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_gradient_preview(
    gradient_stops: list[tuple[float, int, int, int]],
    width: int = 400,
    height: int = 40,
) -> io.BytesIO:
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    ramp_rgb = np.stack([ramp, ramp, ramp], axis=-1)
    ramp_img = Image.fromarray(ramp_rgb, mode="RGB")
    preview = apply_gradient_map(ramp_img, gradient_stops).resize((width, height), Image.NEAREST)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    buf.seek(0)
    return buf


def render_palette_chart(colors: np.ndarray, counts: np.ndarray) -> plt.Figure:
    total = counts.sum()
    percentages = counts / total

    fig, ax = plt.subplots(figsize=(max(8, len(colors) * 1.2), 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a1a")

    x = 0.0
    for rgb, pct in zip(colors, percentages):
        w = float(pct)
        hex_color = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        ax.add_patch(mpatches.Rectangle((x, 0.3), w, 0.6, color=hex_color))
        cx = x + w / 2
        name = nearest_color_name(tuple(int(v) for v in rgb))
        pct_str = f"{pct * 100:.1f}%"
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        txt_color = "white" if lum < 128 else "black"
        ax.text(cx, 0.6, f"{pct_str}\n{hex_color}\n{name}",
                ha="center", va="center", fontsize=6.5,
                color=txt_color, fontweight="bold", wrap=True)
        x += w

    fig.tight_layout(pad=0.5)
    return fig


def render_compare_chart(
    colors_a: np.ndarray, counts_a: np.ndarray, label_a: str,
    colors_b: np.ndarray, counts_b: np.ndarray, label_b: str,
) -> io.BytesIO:
    """Two-row palette comparison chart."""
    fig, axes = plt.subplots(2, 1, figsize=(max(8, max(len(colors_a), len(colors_b)) * 1.2), 5.0))
    fig.patch.set_facecolor("#1a1a1a")

    for ax, colors, counts, label in zip(
        axes, [colors_a, colors_b], [counts_a, counts_b], [label_a, label_b]
    ):
        total = counts.sum()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(label, color="white", fontsize=10, pad=4)
        x = 0.0
        for rgb, cnt in zip(colors, counts):
            w = cnt / total
            hex_color = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
            ax.add_patch(mpatches.Rectangle((x, 0.1), w, 0.8, color=hex_color))
            cx = x + w / 2
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            txt = "white" if lum < 128 else "black"
            ax.text(cx, 0.5, f"{cnt/total*100:.0f}%\n{hex_color}",
                    ha="center", va="center", fontsize=5.5, color=txt, fontweight="bold")
            x += w

    fig.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#1a1a1a")
    plt.close(fig)
    buf.seek(0)
    return buf


def render_harmony_chart(
    orig_colors: list[tuple[int, int, int]],
    suggested: list[tuple[tuple[int, int, int], str]],
) -> io.BytesIO:
    n_orig = len(orig_colors)
    n_sugg = len(suggested)
    n_cols = max(n_orig, n_sugg, 1)

    fig, axes = plt.subplots(2, 1, figsize=(max(5, n_cols * 1.1), 4.5))
    fig.patch.set_facecolor("#1a1a1a")

    def _draw_row(ax, items, title):
        ax.set_xlim(0, len(items))
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=10, pad=4)
        for i, (rgb, lbl) in enumerate(items):
            hex_c = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
            ax.add_patch(mpatches.Rectangle((i + 0.05, 0.15), 0.9, 0.7, color=hex_c))
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            tc = "white" if lum < 128 else "black"
            ax.text(i + 0.5, 0.57, hex_c, ha="center", va="center", fontsize=6.5, color=tc, fontweight="bold")
            ax.text(i + 0.5, 0.25, lbl,   ha="center", va="center", fontsize=5.5, color=tc)

    orig_items = [(tuple(int(v) for v in c), nearest_color_name(tuple(int(v) for v in c))) for c in orig_colors]
    _draw_row(axes[0], orig_items, "Your Palette")
    _draw_row(axes[1], suggested, "Suggested Harmonies")

    fig.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#1a1a1a")
    plt.close(fig)
    buf.seek(0)
    return buf


def render_color_info_swatch(
    main_rgb: tuple[int, int, int],
    harmonies: list[tuple[tuple[int, int, int], str]],
) -> io.BytesIO:
    all_items = [(main_rgb, "Your Color")] + harmonies
    n = len(all_items)

    fig, ax = plt.subplots(figsize=(n * 1.4, 2.2))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, (rgb, label) in enumerate(all_items):
        hex_c = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        if i == 0:
            ax.add_patch(mpatches.Rectangle((i + 0.04, 0.13), 0.92, 0.74, color="white"))
            ax.add_patch(mpatches.Rectangle((i + 0.07, 0.16), 0.86, 0.68, color=hex_c))
        else:
            ax.add_patch(mpatches.Rectangle((i + 0.05, 0.15), 0.9, 0.7, color=hex_c))
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        tc = "white" if lum < 128 else "black"
        ax.text(i + 0.5, 0.53, hex_c,  ha="center", va="center", fontsize=7, color=tc, fontweight="bold")
        ax.text(i + 0.5, 0.09, label,  ha="center", va="bottom",  fontsize=6, color="white")

    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#1a1a1a")
    plt.close(fig)
    buf.seek(0)
    return buf


def render_hue_saturation_chart(img: Image.Image) -> plt.Figure:
    hsv = np.array(img.convert("HSV"))
    hue_deg = hsv[:, :, 0].flatten().astype(float) / 255 * 360
    sat_pct = hsv[:, :, 1].flatten().astype(float) / 255 * 100

    fig = plt.figure(figsize=(10, 4.5), facecolor="#1a1a1a")

    ax_polar = fig.add_subplot(1, 2, 1, polar=True)
    bins = 36
    hue_hist, hue_edges = np.histogram(hue_deg, bins=bins, range=(0, 360))
    theta = np.deg2rad(hue_edges[:-1] + 5)
    width = 2 * math.pi / bins
    bar_colors = [plt.cm.hsv(h / 360) for h in hue_edges[:-1]]
    ax_polar.bar(theta, hue_hist, width=width, bottom=0,
                 color=bar_colors, alpha=0.85, edgecolor="none")
    ax_polar.set_facecolor("#1a1a1a")
    ax_polar.tick_params(colors="white", labelsize=7)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title("Hue Distribution", color="white", pad=12, fontsize=11)
    for label in ax_polar.get_xticklabels():
        label.set_color("white")
    for label in ax_polar.get_yticklabels():
        label.set_color("#888888")

    ax_sat = fig.add_subplot(1, 2, 2)
    ax_sat.set_facecolor("#1a1a1a")
    sat_hist, sat_edges = np.histogram(sat_pct, bins=10, range=(0, 100))
    bar_x = (sat_edges[:-1] + sat_edges[1:]) / 2
    ax_sat.bar(bar_x, sat_hist, width=9, color="#4fc3f7", alpha=0.85, edgecolor="none")
    ax_sat.set_xlabel("Saturation (%)", color="white", fontsize=10)
    ax_sat.set_ylabel("Pixel Count", color="white", fontsize=10)
    ax_sat.set_title("Saturation Distribution", color="white", fontsize=11)
    ax_sat.tick_params(colors="white")
    ax_sat.spines[:].set_color("#444444")

    fig.tight_layout(pad=1.5)
    return fig


def render_chart_to_bytesio(fig: plt.Figure) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def _sanitize_export_name(name: str, *, css: bool = False) -> str:
    """Sanitize a user-supplied export name.

    Strips all control characters so header-injection in line-oriented
    formats (GPL, GGR) is impossible.
    When *css* is True the result is further restricted to ``[a-z0-9-]``
    so it is safe for use in CSS custom-property names.
    """
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    if css:
        name = re.sub(r"[^a-z0-9-]", "", name.lower().replace(" ", "-"))
    return name


def export_ase(colors: list[tuple[int, int, int]], name: str = "Palette") -> bytes:
    blocks = []
    for i, (r, g, b) in enumerate(colors):
        color_name = f"{name} {i + 1}"
        name_utf16 = (color_name + "\x00").encode("utf-16-be")
        name_len = len(name_utf16) // 2
        block_data = struct.pack(">H", name_len)
        block_data += name_utf16
        block_data += b"RGB "
        block_data += struct.pack(">fff", r / 255.0, g / 255.0, b / 255.0)
        block_data += struct.pack(">H", 0)
        blocks.append(struct.pack(">HI", 0x0001, len(block_data)) + block_data)
    return b"ASEF" + struct.pack(">HHI", 1, 0, len(blocks)) + b"".join(blocks)


def export_swatches(colors: list[tuple[int, int, int]], name: str = "Palette") -> bytes:
    swatches = []
    for r, g, b in colors:
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        swatches.append({"hue": round(h, 6), "saturation": round(s, 6),
                         "brightness": round(v, 6), "alpha": 1.0, "colorSpace": 0})
    while len(swatches) < 30:
        swatches.append(None)
    return json.dumps({"name": name, "swatches": swatches}).encode("utf-8")


def export_gpl(colors: list[tuple[int, int, int]], name: str = "Palette") -> bytes:
    """GIMP Palette (.gpl) format."""
    name = _sanitize_export_name(name)
    lines = ["GIMP Palette", f"Name: {name}", "Columns: 0", "#"]
    for r, g, b in colors:
        color_name = nearest_color_name((r, g, b))
        lines.append(f"{r:3d} {g:3d} {b:3d}\t{color_name}")
    return "\n".join(lines).encode("utf-8")


def export_aco(colors: list[tuple[int, int, int]], name: str = "Palette") -> bytes:
    """Photoshop Color Swatch (.aco) — version 1 + version 2 block."""
    n = len(colors)

    # Version 1
    data = struct.pack(">HH", 1, n)
    for r, g, b in colors:
        data += struct.pack(">HHHHH", 0, r * 257, g * 257, b * 257, 0)

    # Version 2 (same + Unicode names)
    data += struct.pack(">HH", 2, n)
    for i, (r, g, b) in enumerate(colors):
        data += struct.pack(">HHHHH", 0, r * 257, g * 257, b * 257, 0)
        color_name = f"{name} {i + 1}\x00"   # null-terminated
        name_utf16 = color_name.encode("utf-16-be")
        data += struct.pack(">HH", 0, len(color_name))  # zero mark, char count incl. null
        data += name_utf16

    return data


def export_css(colors: list[tuple[int, int, int]], name: str = "palette") -> bytes:
    """CSS custom properties (.css)."""
    prefix = _sanitize_export_name(name, css=True)
    lines = [":root {"]
    for i, (r, g, b) in enumerate(colors):
        hex_val = f"#{r:02X}{g:02X}{b:02X}"
        lines.append(f"  --{prefix}-{i + 1}: {hex_val};")
        lines.append(f"  --{prefix}-{i + 1}-rgb: {r}, {g}, {b};")
    lines.append("}")
    return "\n".join(lines).encode("utf-8")


def export_tailwind(colors: list[tuple[int, int, int]], name: str = "palette") -> bytes:
    """Tailwind config color extension (.json)."""
    prefix = _sanitize_export_name(name, css=True)
    color_dict = {
        f"{prefix}-{i + 1}": f"#{r:02X}{g:02X}{b:02X}"
        for i, (r, g, b) in enumerate(colors)
    }
    config = {"theme": {"extend": {"colors": color_dict}}}
    return json.dumps(config, indent=2).encode("utf-8")


def export_gradient_ggr(
    gradient_stops: list[tuple[float, int, int, int]],
    name: str = "palette_gradient",
) -> bytes:
    name = _sanitize_export_name(name)
    stops = sorted(gradient_stops, key=lambda s: s[0])
    n_segments = len(stops) - 1
    lines = ["GIMP Gradient", f"Name: {name}", str(n_segments)]
    for i in range(n_segments):
        left_pos, r0, g0, b0 = stops[i]
        right_pos, r1, g1, b1 = stops[i + 1]
        mid = (left_pos + right_pos) / 2.0
        lines.append(
            f"{left_pos:.6f} {mid:.6f} {right_pos:.6f} "
            f"{r0/255:.6f} {g0/255:.6f} {b0/255:.6f} 1.000000 "
            f"{r1/255:.6f} {g1/255:.6f} {b1/255:.6f} 1.000000 0 0"
        )
    return "\n".join(lines).encode("utf-8")


def export_gradient_json(
    gradient_stops: list[tuple[float, int, int, int]],
    name: str = "palette_gradient",
) -> bytes:
    name = _sanitize_export_name(name)
    stops_list = [
        {"position": round(pos, 6), "r": r, "g": g, "b": b}
        for pos, r, g, b in sorted(gradient_stops, key=lambda s: s[0])
    ]
    return json.dumps({"name": name, "stops": stops_list}, indent=2).encode("utf-8")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python analyzer.py <image_path>")
        sys.exit(1)

    with open(path, "rb") as f:
        data = f.read()

    img = load_image_from_bytes(data)
    colors, counts = extract_dominant_colors(img, n=8)
    stats = compute_stats(img)

    print(f"Image: {stats['width']}x{stats['height']}")
    print(f"Brightness: {stats['brightness']}")
    print(f"Contrast:   {stats['contrast']}")
    print(f"Saturation: {stats['mean_saturation_pct']}%")
    hue_range = stats['dominant_hue_range']
    if hue_range == (None, None):
        print("Dominant hue range: no chromatic pixels")
    else:
        print(f"Dominant hue range: {hue_range[0]}-{hue_range[1]} deg")
    print("\nDominant colors:")
    total = counts.sum()
    for rgb, cnt in zip(colors, counts):
        hex_c = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        name = nearest_color_name(tuple(int(v) for v in rgb))
        pct = cnt / total * 100
        print(f"  {hex_c}  {name:20s}  {pct:.1f}%")
