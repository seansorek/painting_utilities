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


def _build_stats_embed(
    stats: dict,
    colors,
    counts,
    image_name: str,
    grayscale_warning: bool,
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
    embed.add_field(name="Contrast", value=f"{stats['contrast']}", inline=True)
    embed.add_field(name="Mean Saturation", value=f"{stats['mean_saturation_pct']}%", inline=True)
    embed.add_field(
        name="Dominant Hue Range",
        value=_hue_range_label(stats["dominant_hue_range"]),
        inline=True,
    )

    top_colors_text = "\n".join(
        f"`#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}` "
        f"**{nearest_color_name(tuple(int(v) for v in rgb))}** "
        f"— {cnt / total * 100:.1f}%"
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
        description="Number of dominant colors to extract (default 8)",
        default=8,
        min_value=3,
        max_value=16,
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

        embed = _build_stats_embed(stats, colors, counts, image.filename, grayscale_warning)

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
        description="Number of dominant colors to extract (default 8)",
        default=8,
        min_value=3,
        max_value=16,
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
            f"`#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}` "
            f"**{nearest_color_name(tuple(int(v) for v in rgb))}** — {cnt / total * 100:.1f}%"
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


if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN not set. Copy .env.example to .env and fill it in.")
    bot.run(TOKEN)
