import io
import json
import math
import struct
import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Named color reference table: (name, R, G, B)
_NAMED_COLORS = [
    ("Red",       220,  20,  60),
    ("Orange",    255, 140,   0),
    ("Yellow",    255, 215,   0),
    ("Lime",      50,  205,  50),
    ("Green",      0, 128,   0),
    ("Teal",       0, 128, 128),
    ("Cyan",       0, 200, 200),
    ("Sky Blue",   30, 144, 255),
    ("Blue",       30,  30, 220),
    ("Indigo",     75,   0, 130),
    ("Violet",    148,   0, 211),
    ("Magenta",   220,   0, 180),
    ("Pink",      255, 105, 180),
    ("Brown",     139,  69,  19),
    ("Tan",       210, 180, 140),
    ("White",     255, 255, 255),
    ("Light Gray",192, 192, 192),
    ("Gray",      128, 128, 128),
    ("Dark Gray",  64,  64,  64),
    ("Black",       0,   0,   0),
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

    # Maximum pairwise angular spread
    max_spread = max(_circ_dist(a, b) for i, a in enumerate(saturated_hues)
                     for b in saturated_hues[i + 1:])

    if max_spread < 30:
        return "Monochromatic"
    if max_spread < 60:
        return "Analogous"

    # Complementary: any pair ~180° apart
    for i, a in enumerate(saturated_hues):
        for b in saturated_hues[i + 1:]:
            if abs(_circ_dist(a, b) - 180) <= 30:
                # Split complementary: third color ~150° from one end
                others = [h for h in saturated_hues if h != a and h != b]
                for c in others:
                    if abs(_circ_dist(a, c) - 150) <= 30 or abs(_circ_dist(b, c) - 150) <= 30:
                        return "Split Complementary"
                return "Complementary"

    # Triadic: three colors each ~120° apart
    hues = sorted(saturated_hues)
    for i, a in enumerate(hues):
        for b in hues[i + 1:]:
            if abs(_circ_dist(a, b) - 120) <= 30:
                for c in hues:
                    if c != a and c != b:
                        if abs(_circ_dist(b, c) - 120) <= 30 and abs(_circ_dist(a, c) - 120) <= 30:
                            return "Triadic"

    return "Polychromatic"


def palette_to_gradient_stops(
    colors: np.ndarray, counts: np.ndarray
) -> list[tuple[float, int, int, int]]:
    lums = np.array([0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in colors])
    order = np.argsort(lums)
    sorted_colors = colors[order]
    n = len(sorted_colors)
    stops = []
    for i, rgb in enumerate(sorted_colors):
        pos = i / max(n - 1, 1)
        stops.append((pos, int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return stops


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


def load_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB")


def extract_dominant_colors(
    img: Image.Image, n: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    small = img.resize((200, 200), Image.LANCZOS)
    pixels = np.array(small).reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    km.fit(pixels)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(km.labels_, minlength=n)
    order = np.argsort(counts)[::-1]
    return centers[order], counts[order]


def compute_stats(img: Image.Image) -> dict:
    arr = np.array(img).astype(float)
    brightness = arr.mean()
    # Luminance per pixel via Rec.601 weighting
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    contrast = lum.std()

    # Hue and saturation via HSV conversion
    hsv = np.array(img.convert("HSV"))
    hue_vals = hsv[:, :, 0].flatten()      # 0-255 mapped to 0-360
    sat_vals = hsv[:, :, 1].flatten()      # 0-255

    mean_sat_pct = sat_vals.mean() / 255 * 100

    # Find dominant hue range (highest-density 60-degree window)
    hue_deg = hue_vals.astype(float) / 255 * 360
    hist, edges = np.histogram(hue_deg, bins=36, range=(0, 360))
    # Rolling sum for 60-degree windows (6 bins)
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


def render_palette_chart(
    colors: np.ndarray, counts: np.ndarray
) -> plt.Figure:
    total = counts.sum()
    percentages = counts / total

    fig, ax = plt.subplots(figsize=(max(8, len(colors) * 1.2), 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a1a")

    x = 0.0
    swatch_height = 0.6
    label_y = 0.58

    for i, (rgb, pct) in enumerate(zip(colors, percentages)):
        w = float(pct)
        hex_color = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        ax.add_patch(mpatches.Rectangle((x, 0.3), w, swatch_height, color=hex_color))

        cx = x + w / 2
        name = nearest_color_name(tuple(int(v) for v in rgb))
        pct_str = f"{pct * 100:.1f}%"

        # Choose white or black label based on luminance
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        txt_color = "white" if lum < 128 else "black"

        ax.text(cx, 0.6, f"{pct_str}\n{hex_color}\n{name}",
                ha="center", va="center", fontsize=6.5,
                color=txt_color, fontweight="bold",
                wrap=True)
        x += w

    fig.tight_layout(pad=0.5)
    return fig


def render_hue_saturation_chart(img: Image.Image) -> plt.Figure:
    hsv = np.array(img.convert("HSV"))
    hue_deg = hsv[:, :, 0].flatten().astype(float) / 255 * 360
    sat_pct = hsv[:, :, 1].flatten().astype(float) / 255 * 100

    fig = plt.figure(figsize=(10, 4.5), facecolor="#1a1a1a")

    # --- Polar hue histogram ---
    ax_polar = fig.add_subplot(1, 2, 1, polar=True)
    bins = 36
    hue_hist, hue_edges = np.histogram(hue_deg, bins=bins, range=(0, 360))
    theta = np.deg2rad(hue_edges[:-1] + 5)  # center of each bin
    width = 2 * math.pi / bins

    # Color each bar by its hue
    bar_colors = [
        plt.cm.hsv(h / 360) for h in hue_edges[:-1]
    ]
    bars = ax_polar.bar(theta, hue_hist, width=width, bottom=0,
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

    # --- Saturation histogram ---
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


# --- Standalone test ---
if __name__ == "__main__":
    import sys, os

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
    print(f"Dominant hue range: {stats['dominant_hue_range'][0]}-{stats['dominant_hue_range'][1]} deg")
    print("\nDominant colors:")
    total = counts.sum()
    for rgb, cnt in zip(colors, counts):
        hex_c = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        name = nearest_color_name(tuple(int(v) for v in rgb))
        pct = cnt / total * 100
        print(f"  {hex_c}  {name:12s}  {pct:.1f}%")

    palette_fig = render_palette_chart(colors, counts)
    hue_sat_fig = render_hue_saturation_chart(img)

    palette_path = "test_palette.png"
    hue_path = "test_hue_sat.png"
    render_chart_to_bytesio(palette_fig).read()  # discard — save via savefig instead

    palette_fig2 = render_palette_chart(colors, counts)
    palette_fig2.savefig(palette_path, bbox_inches="tight", dpi=150, facecolor="#1a1a1a")
    plt.close(palette_fig2)

    hue_sat_fig2 = render_hue_saturation_chart(img)
    hue_sat_fig2.savefig(hue_path, bbox_inches="tight", dpi=150, facecolor="#1a1a1a")
    plt.close(hue_sat_fig2)

    print(f"\nSaved: {palette_path}, {hue_path}")
