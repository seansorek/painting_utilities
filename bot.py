import os
import traceback

import discord
from dotenv import load_dotenv

import io

from analyzer import (
    load_image_from_bytes,
    extract_dominant_colors,
    compute_stats,
    render_palette_chart,
    render_hue_saturation_chart,
    render_chart_to_bytesio,
    nearest_color_name,
    apply_gradient_map,
    GRADIENT_PRESETS,
    parse_hex_color,
    render_gradient_preview,
    rgb_to_cmyk,
    classify_palette_type,
    palette_to_gradient_stops,
    export_ase,
    export_swatches,
)

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("DISCORD_GUILD_ID")
guild_ids = [int(GUILD_ID)] if GUILD_ID else None

MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB

bot = discord.Bot(intents=discord.Intents.default())


@bot.event
async def on_ready():
    print(f"Ready — logged in as {bot.user} (id: {bot.user.id})")


def _hue_range_label(hue_range: tuple[int, int]) -> str:
    hue_names = {
        (0, 30): "Red", (30, 60): "Orange", (60, 90): "Yellow",
        (90, 150): "Green", (150, 210): "Cyan", (210, 270): "Blue",
        (270, 330): "Violet", (330, 360): "Red",
    }
    start, end = hue_range
    for (lo, hi), name in hue_names.items():
        mid = (start + end) / 2 % 360
        if lo <= mid < hi:
            return f"{name} ({start}–{end}°)"
    return f"{start}–{end}°"


def _contrast_level(contrast: float) -> str:
    if contrast < 30:
        label = "Low"
    elif contrast < 70:
        label = "Medium"
    else:
        label = "High"
    return f"{label} ({contrast})"


def _saturation_level(sat_pct: float) -> str:
    if sat_pct < 20:
        label = "Low"
    elif sat_pct < 60:
        label = "Medium"
    else:
        label = "High"
    return f"{label} ({sat_pct}%)"


def _color_line(rgb, cnt, total, show_rgb: bool, show_cmyk: bool) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    hex_str = f"#{r:02X}{g:02X}{b:02X}"
    name = nearest_color_name((r, g, b))
    pct = cnt / total * 100
    line = f"`{hex_str}` **{name}** — {pct:.1f}%"
    if show_rgb:
        line += f"\n  RGB({r}, {g}, {b})"
    if show_cmyk:
        c, m, y, k = rgb_to_cmyk(r, g, b)
        line += f"\n  CMYK({c}%, {m}%, {y}%, {k}%)"
    return line


def _build_stats_embed(
    stats: dict,
    colors,
    counts,
    image_name: str,
    grayscale_warning: bool,
    show_rgb: bool = False,
    show_cmyk: bool = False,
) -> discord.Embed:
    total = counts.sum()
    embed = discord.Embed(
        title=f"Image Analysis — {image_name}",
        color=discord.Color.from_rgb(
            int(colors[0][0]), int(colors[0][1]), int(colors[0][2])
        ),
    )
    embed.add_field(name="Dimensions", value=f"{stats['width']} × {stats['height']} px", inline=True)
    embed.add_field(name="Brightness", value=f"{stats['brightness']} / 255", inline=True)
    embed.add_field(name="Contrast", value=_contrast_level(stats["contrast"]), inline=True)
    embed.add_field(name="Mean Saturation", value=_saturation_level(stats["mean_saturation_pct"]), inline=True)
    embed.add_field(
        name="Dominant Hue Range",
        value=_hue_range_label(stats["dominant_hue_range"]),
        inline=True,
    )
    palette_type = classify_palette_type(colors, counts)
    embed.add_field(name="Palette Type", value=palette_type, inline=True)

    top_colors_text = "\n".join(
        _color_line(rgb, cnt, total, show_rgb, show_cmyk)
        for rgb, cnt in zip(colors[:5], counts[:5])
    )
    embed.add_field(name="Top Colors", value=top_colors_text, inline=False)

    if grayscale_warning:
        embed.add_field(
            name="Note",
            value="Mean saturation is below 15% — this image is mostly grayscale.",
            inline=False,
        )

    embed.set_image(url="attachment://palette.png")
    return embed


@bot.slash_command(
    name="analyze",
    description="Full analysis: dominant colors, stats, and charts",
    guild_ids=guild_ids,
)
async def analyze(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Upload a painting image"),
    num_colors: discord.Option(
        int,
        description="Number of dominant colors to extract (default 10)",
        default=10,
        min_value=3,
        max_value=16,
    ),
    show_rgb: discord.Option(
        bool,
        description="Show RGB values for each color",
        default=False,
        required=False,
    ),
    show_cmyk: discord.Option(
        bool,
        description="Show CMYK values for each color",
        default=False,
        required=False,
    ),
):
    await ctx.defer()

    if not image.content_type or not image.content_type.startswith("image/"):
        await ctx.followup.send("Please attach an image file (PNG, JPEG, etc.).")
        return

    if image.size > MAX_FILE_BYTES:
        await ctx.followup.send(f"Image is too large (max 15 MB). Yours is {image.size / 1e6:.1f} MB.")
        return

    try:
        data = await image.read()
        img = load_image_from_bytes(data)
        colors, counts = extract_dominant_colors(img, n=num_colors)
        stats = compute_stats(img)

        grayscale_warning = stats["mean_saturation_pct"] < 15

        palette_buf = render_chart_to_bytesio(render_palette_chart(colors, counts))
        hue_sat_buf = render_chart_to_bytesio(render_hue_saturation_chart(img))

        embed = _build_stats_embed(stats, colors, counts, image.filename, grayscale_warning, show_rgb, show_cmyk)

        await ctx.followup.send(
            embed=embed,
            files=[
                discord.File(palette_buf, filename="palette.png"),
                discord.File(hue_sat_buf, filename="hue_sat.png"),
            ],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong analyzing that image. Make sure it's a valid image file.")


@bot.slash_command(
    name="palette",
    description="Quick color palette swatch only",
    guild_ids=guild_ids,
)
async def palette(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Upload a painting image"),
    num_colors: discord.Option(
        int,
        description="Number of dominant colors to extract (default 10)",
        default=10,
        min_value=3,
        max_value=16,
    ),
    show_rgb: discord.Option(
        bool,
        description="Show RGB values for each color",
        default=False,
        required=False,
    ),
    show_cmyk: discord.Option(
        bool,
        description="Show CMYK values for each color",
        default=False,
        required=False,
    ),
):
    await ctx.defer()

    if not image.content_type or not image.content_type.startswith("image/"):
        await ctx.followup.send("Please attach an image file (PNG, JPEG, etc.).")
        return

    if image.size > MAX_FILE_BYTES:
        await ctx.followup.send(f"Image is too large (max 15 MB). Yours is {image.size / 1e6:.1f} MB.")
        return

    try:
        data = await image.read()
        img = load_image_from_bytes(data)
        colors, counts = extract_dominant_colors(img, n=num_colors)
        palette_buf = render_chart_to_bytesio(render_palette_chart(colors, counts))

        total = counts.sum()
        lines = [
            _color_line(rgb, cnt, total, show_rgb, show_cmyk)
            for rgb, cnt in zip(colors, counts)
        ]
        embed = discord.Embed(
            title=f"Color Palette — {image.filename}",
            description="\n".join(lines),
            color=discord.Color.from_rgb(
                int(colors[0][0]), int(colors[0][1]), int(colors[0][2])
            ),
        )
        embed.set_image(url="attachment://palette.png")

        await ctx.followup.send(
            embed=embed,
            files=[discord.File(palette_buf, filename="palette.png")],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong. Make sure it's a valid image file.")


def _build_gradient_embed(
    filename: str,
    gradient_label: str,
    img,
    gradient_stops: list,
) -> discord.Embed:
    _, r0, g0, b0 = gradient_stops[0]
    _, r1, g1, b1 = gradient_stops[-1]
    mid_r, mid_g, mid_b = (r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2
    embed = discord.Embed(
        title=f"Gradient Map — {filename}",
        color=discord.Color.from_rgb(mid_r, mid_g, mid_b),
    )
    embed.add_field(name="Gradient", value=gradient_label, inline=True)
    embed.add_field(name="Dimensions", value=f"{img.width} × {img.height} px", inline=True)
    embed.set_image(url="attachment://gradient_map.png")
    embed.set_thumbnail(url="attachment://gradient_swatch.png")
    return embed


@bot.slash_command(
    name="gradient_map",
    description="Remap image tones through a color gradient",
    guild_ids=guild_ids,
)
async def gradient_map_cmd(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Image to process", required=True),
    preset: discord.Option(
        str,
        description="Predefined gradient (default: fire)",
        choices=["fire", "ocean", "forest", "amethyst", "grayscale", "sunset", "ice"],
        required=False,
        default="fire",
    ),
    start_color: discord.Option(
        str,
        description="Custom shadow color as hex (e.g. #1a0030) — overrides preset if both given",
        required=False,
        default=None,
    ),
    end_color: discord.Option(
        str,
        description="Custom highlight color as hex (e.g. #ffe080) — overrides preset if both given",
        required=False,
        default=None,
    ),
):
    await ctx.defer()

    if not image.content_type or not image.content_type.startswith("image/"):
        await ctx.followup.send("Please attach an image file (PNG, JPEG, etc.).")
        return

    if image.size > MAX_FILE_BYTES:
        await ctx.followup.send(f"Image is too large (max 15 MB). Yours is {image.size / 1e6:.1f} MB.")
        return

    if (start_color is None) != (end_color is None):
        await ctx.followup.send("Provide both `start_color` and `end_color`, or neither.")
        return

    gradient_stops = None
    gradient_label = preset

    if start_color is not None:
        try:
            r0, g0, b0 = parse_hex_color(start_color)
            r1, g1, b1 = parse_hex_color(end_color)
        except ValueError as e:
            await ctx.followup.send(str(e))
            return
        gradient_stops = [(0.0, r0, g0, b0), (1.0, r1, g1, b1)]
        gradient_label = f"custom ({start_color.upper()} → {end_color.upper()})"
    else:
        gradient_stops = GRADIENT_PRESETS[preset]

    try:
        data = await image.read()
        img = load_image_from_bytes(data)
        result = apply_gradient_map(img, gradient_stops)

        out_buf = io.BytesIO()
        result.save(out_buf, format="PNG")
        out_buf.seek(0)

        swatch_buf = render_gradient_preview(gradient_stops)
        embed = _build_gradient_embed(image.filename, gradient_label, img, gradient_stops)

        await ctx.followup.send(
            embed=embed,
            files=[
                discord.File(out_buf, filename="gradient_map.png"),
                discord.File(swatch_buf, filename="gradient_swatch.png"),
            ],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong processing the image.")


@bot.slash_command(
    name="palette_gradient",
    description="Generate a gradient from the image's own colors and apply it as a tone map",
    guild_ids=guild_ids,
)
async def palette_gradient_cmd(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Image to process", required=True),
    num_colors: discord.Option(
        int,
        description="Number of colors to extract for the gradient (default 5)",
        default=5,
        min_value=3,
        max_value=10,
    ),
):
    await ctx.defer()

    if not image.content_type or not image.content_type.startswith("image/"):
        await ctx.followup.send("Please attach an image file (PNG, JPEG, etc.).")
        return

    if image.size > MAX_FILE_BYTES:
        await ctx.followup.send(f"Image is too large (max 15 MB). Yours is {image.size / 1e6:.1f} MB.")
        return

    try:
        data = await image.read()
        img = load_image_from_bytes(data)
        colors, counts = extract_dominant_colors(img, n=num_colors)
        gradient_stops = palette_to_gradient_stops(colors, counts)

        result = apply_gradient_map(img, gradient_stops)
        out_buf = io.BytesIO()
        result.save(out_buf, format="PNG")
        out_buf.seek(0)

        swatch_buf = render_gradient_preview(gradient_stops)

        stop_colors = " → ".join(
            f"#{r:02X}{g:02X}{b:02X}" for _, r, g, b in gradient_stops
        )
        embed = discord.Embed(
            title=f"Palette Gradient — {image.filename}",
            color=discord.Color.from_rgb(*[int(v) for v in colors[0]]),
        )
        embed.add_field(name="Gradient", value=stop_colors, inline=False)
        embed.add_field(name="Dimensions", value=f"{img.width} × {img.height} px", inline=True)
        embed.set_image(url="attachment://gradient_map.png")
        embed.set_thumbnail(url="attachment://gradient_swatch.png")

        await ctx.followup.send(
            embed=embed,
            files=[
                discord.File(out_buf, filename="gradient_map.png"),
                discord.File(swatch_buf, filename="gradient_swatch.png"),
            ],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong processing the image.")


@bot.slash_command(
    name="export_palette",
    description="Export dominant colors as an .ase (Photoshop) or .swatches (Procreate) file",
    guild_ids=guild_ids,
)
async def export_palette_cmd(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Image to extract colors from", required=True),
    format: discord.Option(
        str,
        description="Export format",
        choices=["ase", "swatches"],
        default="ase",
    ),
    num_colors: discord.Option(
        int,
        description="Number of colors to extract (default 10)",
        default=10,
        min_value=3,
        max_value=16,
    ),
    palette_name: discord.Option(
        str,
        description="Name for the palette (default: Palette)",
        default="Palette",
        required=False,
    ),
):
    await ctx.defer()

    if not image.content_type or not image.content_type.startswith("image/"):
        await ctx.followup.send("Please attach an image file (PNG, JPEG, etc.).")
        return

    if image.size > MAX_FILE_BYTES:
        await ctx.followup.send(f"Image is too large (max 15 MB). Yours is {image.size / 1e6:.1f} MB.")
        return

    try:
        data = await image.read()
        img = load_image_from_bytes(data)
        colors, counts = extract_dominant_colors(img, n=num_colors)

        color_list = [(int(c[0]), int(c[1]), int(c[2])) for c in colors]

        if format == "ase":
            file_bytes = export_ase(color_list, palette_name)
            filename = "palette.ase"
        else:
            file_bytes = export_swatches(color_list, palette_name)
            filename = "palette.swatches"

        total = counts.sum()
        lines = [
            f"`#{r:02X}{g:02X}{b:02X}` **{nearest_color_name((r, g, b))}** — {cnt / total * 100:.1f}%"
            for (r, g, b), cnt in zip(color_list, counts)
        ]
        embed = discord.Embed(
            title=f"Palette Export — {image.filename}",
            description="\n".join(lines),
            color=discord.Color.from_rgb(*color_list[0]),
        )
        embed.add_field(name="Format", value=f".{format}", inline=True)
        embed.add_field(name="Colors", value=str(num_colors), inline=True)

        await ctx.followup.send(
            embed=embed,
            files=[discord.File(io.BytesIO(file_bytes), filename=filename)],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong exporting the palette.")


COMMAND_DOCS = {
    "analyze": {
        "summary": "Full analysis: dominant colors, image stats, and two charts",
        "description": (
            "Uploads a painting and runs a full analysis: extracts dominant colors via KMeans clustering, "
            "computes image statistics (brightness, contrast, saturation, dominant hue range, palette type), "
            "and generates two chart images — a color palette swatch and a hue/saturation distribution chart."
        ),
        "params": [
            ("`image`", "required", "The painting to analyze (PNG, JPEG, etc., max 15 MB)"),
            ("`num_colors`", "3–16, default 10", "How many dominant colors to extract"),
            ("`show_rgb`", "true/false, default false", "Include RGB values alongside each color"),
            ("`show_cmyk`", "true/false, default false", "Include CMYK values alongside each color"),
        ],
        "output": "Embed with stats + top 5 colors, attached `palette.png` swatch chart and `hue_sat.png` distribution chart",
    },
    "palette": {
        "summary": "Quick color palette swatch — no stats, just colors",
        "description": (
            "Extracts dominant colors from the image and lists all of them with hex codes, color names, "
            "and percentage coverage. Faster than /analyze when you only need the palette."
        ),
        "params": [
            ("`image`", "required", "The painting to analyze (PNG, JPEG, etc., max 15 MB)"),
            ("`num_colors`", "3–16, default 10", "How many dominant colors to extract"),
            ("`show_rgb`", "true/false, default false", "Include RGB values alongside each color"),
            ("`show_cmyk`", "true/false, default false", "Include CMYK values alongside each color"),
        ],
        "output": "Embed listing all colors with hex/name/%, attached `palette.png` swatch chart",
    },
    "gradient_map": {
        "summary": "Remap image tones through a color gradient",
        "description": (
            "Applies a gradient map to the image — each pixel's luminance value is remapped to a color "
            "from the chosen gradient, replacing the original colors while preserving light/dark structure. "
            "Choose a preset or supply custom shadow/highlight hex colors."
        ),
        "params": [
            ("`image`", "required", "Image to process (PNG, JPEG, etc., max 15 MB)"),
            ("`preset`", "fire/ocean/forest/amethyst/grayscale/sunset/ice, default fire", "Built-in gradient preset"),
            ("`start_color`", "hex, e.g. `#1a0030`, optional", "Custom shadow (darkest) color — must be paired with `end_color`"),
            ("`end_color`", "hex, e.g. `#ffe080`, optional", "Custom highlight (lightest) color — must be paired with `start_color`"),
        ],
        "output": "Embed with gradient label and dimensions, attached `gradient_map.png` result and `gradient_swatch.png` preview",
    },
    "palette_gradient": {
        "summary": "Auto-generate a gradient from the image's own colors and apply it",
        "description": (
            "Extracts the dominant colors from the image, orders them by luminance to form gradient stops, "
            "then applies that gradient as a tone map back onto the image. No preset needed — the gradient "
            "is derived entirely from the image itself."
        ),
        "params": [
            ("`image`", "required", "Image to process (PNG, JPEG, etc., max 15 MB)"),
            ("`num_colors`", "3–10, default 5", "Number of colors to extract for the gradient"),
        ],
        "output": "Embed with gradient hex stops and dimensions, attached `gradient_map.png` result and `gradient_swatch.png` preview",
    },
    "export_palette": {
        "summary": "Export colors as an .ase (Photoshop) or .swatches (Procreate) file",
        "description": (
            "Extracts dominant colors and exports them as a palette file compatible with design software. "
            "`ase` produces an Adobe Swatch Exchange file importable in Photoshop, Illustrator, and InDesign. "
            "`swatches` produces a JSON-based file importable directly in Procreate."
        ),
        "params": [
            ("`image`", "required", "Image to extract colors from (PNG, JPEG, etc., max 15 MB)"),
            ("`format`", "ase/swatches, default ase", "Export format: `.ase` for Adobe apps or `.swatches` for Procreate"),
            ("`num_colors`", "3–16, default 10", "Number of colors to extract"),
            ("`palette_name`", "text, default `Palette`", "Name embedded in the palette file"),
        ],
        "output": "Embed listing all colors, attached `palette.ase` or `palette.swatches` file for download",
    },
}

_HELP_CHOICES = list(COMMAND_DOCS.keys())


@bot.slash_command(
    name="help",
    description="Show available commands and how to use them",
    guild_ids=guild_ids,
)
async def help_cmd(
    ctx: discord.ApplicationContext,
    command: discord.Option(
        str,
        description="Get detailed help for a specific command",
        choices=_HELP_CHOICES,
        required=False,
        default=None,
    ),
):
    if command is None:
        embed = discord.Embed(
            title="Painting Utilities — Commands",
            description="Tools for analyzing painting images and extracting color palettes.",
            color=discord.Color.blurple(),
        )
        for name, doc in COMMAND_DOCS.items():
            embed.add_field(name=f"/{name}", value=doc["summary"], inline=False)
        embed.set_footer(text="Use /help command:<name> for detailed usage and parameters.")
        await ctx.respond(embed=embed)
    else:
        doc = COMMAND_DOCS[command]
        embed = discord.Embed(
            title=f"/{command}",
            description=doc["description"],
            color=discord.Color.blurple(),
        )
        param_lines = "\n".join(
            f"**{p}** ({default}) — {desc}" for p, default, desc in doc["params"]
        )
        embed.add_field(name="Parameters", value=param_lines, inline=False)
        embed.add_field(name="Output", value=doc["output"], inline=False)
        await ctx.respond(embed=embed)


if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN not set. Copy .env.example to .env and fill it in.")
    bot.run(TOKEN)
