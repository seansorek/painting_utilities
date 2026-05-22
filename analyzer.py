import io
import math

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


def nearest_color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    best, best_dist = _NAMED_COLORS[0][0], float("inf")
    for name, nr, ng, nb in _NAMED_COLORS:
        d = (r - nr) ** 2 + (g - ng) ** 2 + (b - nb) ** 2
        if d < best_dist:
            best_dist, best = d, name
    return best


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
