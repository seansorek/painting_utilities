import asyncio
import functools
import hashlib
import io
import json
import os
import random
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date as _date, datetime, timedelta

import discord
import pytz
from discord.ext import commands, tasks
from dotenv import load_dotenv

from analyzer import (
    MAX_IMAGE_PIXELS,
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
    parse_multi_hex_gradient,
    reverse_gradient,
    render_gradient_preview,
    rgb_to_cmyk,
    classify_palette_type,
    palette_to_gradient_stops,
    adjust_image,
    simulate_colorblindness,
    render_colorblind_comparison,
    recolor_image,
    suggest_harmony_colors,
    render_harmony_chart,
    render_color_info_swatch,
    render_compare_chart,
    export_ase,
    export_swatches,
    export_gpl,
    export_aco,
    export_css,
    export_tailwind,
    export_gradient_ggr,
    export_gradient_json,
)

load_dotenv()
TOKEN    = os.getenv("DISCORD_TOKEN")


def _parse_dev_guild_ids(raw: str | None) -> list[int] | None:
    """Parse DISCORD_GUILD_ID.

    DISCORD_GUILD_ID is a *development-only* fast-sync override: when set, slash
    commands register to that single guild and appear instantly. For public
    (multi-server) use it must be left UNSET so commands register globally and
    show up in every server the bot is invited to. A malformed value must not
    crash startup — we warn and fall back to global registration.
    """
    if not raw:
        return None
    try:
        return [int(raw)]
    except ValueError:
        print(
            f"Warning: DISCORD_GUILD_ID={raw!r} is not a valid integer; "
            "ignoring and registering commands globally."
        )
        return None


GUILD_ID = os.getenv("DISCORD_GUILD_ID")
guild_ids = _parse_dev_guild_ids(GUILD_ID)

ET = pytz.timezone("America/New_York")

_REFERENCES_FILE = os.path.join(os.path.dirname(__file__), "references.json")
_SCHEDULE_FILE   = os.path.join(os.path.dirname(__file__), "daily_schedule.json")
_CONFIG_FILE     = os.path.join(os.path.dirname(__file__), "config.json")

MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB

# How many hours past post_at a scheduled challenge is kept for retry before
# being dropped.  Prevents stale entries from accumulating if the bot is down
# for an extended period or a guild channel is permanently misconfigured.
_CHALLENGE_EXPIRY_HOURS = 24

# message_content is intentionally NOT enabled: it is a privileged intent (needs
# Discord approval + verification at 100 servers) and nothing here reads message
# text — on_message only inspects attachments, which arrive without the intent.
_intents = discord.Intents.default()
bot = discord.Bot(intents=_intents)

# ---------------------------------------------------------------------------
# CPU offload + concurrency / rate limiting
#
# All heavy image work (PIL decode, KMeans, numpy, matplotlib) is blocking. A
# single dedicated worker thread keeps the event loop responsive (so the bot
# stays live for every other server while one image is processed) AND confines
# all matplotlib/pyplot access to one thread (pyplot's global state is not
# thread-safe). max_workers=1 therefore also bounds concurrent heavy jobs to one
# process-wide — a natural throttle against abuse.
# ---------------------------------------------------------------------------
_CPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="img-worker")

# Per-user cooldown on expensive commands (seconds). Configurable via env.
_COOLDOWN_SECONDS = float(os.getenv("COMMAND_COOLDOWN_SECONDS", "5"))
_USER_COOLDOWNS: dict[int, float] = {}

# Commands subject to CPU offload + cooldown (the image/render-heavy ones).
_HEAVY_COMMANDS = frozenset({
    "analyze", "palette", "gradient_map", "palette_gradient",
    "export_palette", "export_gradient", "color_info", "compare",
    "colorblind", "recolor", "suggest_harmony",
})


async def _run_cpu(func, *args, **kwargs):
    """Run a blocking CPU-bound callable on the dedicated image worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _CPU_EXECUTOR, functools.partial(func, *args, **kwargs)
    )

_run_sync = _run_cpu


class _CooldownError(commands.CheckFailure):
    """Raised when a user invokes a heavy command before their cooldown expires."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"On cooldown for {retry_after:.1f}s")

# ---------------------------------------------------------------------------
# Result cache — keyed by (sha1(image_bytes), n_colors, sat_boost, bri_boost)
# Stores (colors, counts, stats) so repeated uploads skip KMeans.
# ---------------------------------------------------------------------------
_IMAGE_CACHE: dict[str, tuple] = {}
_CACHE_MAX = 50


def _cache_key(data: bytes, n: int, sat: float = 0.0, bri: float = 0.0) -> str:
    h = hashlib.sha1(data, usedforsecurity=False).hexdigest()
    return f"{h}:{n}:{sat:.3f}:{bri:.3f}"


def _cache_get(data: bytes, n: int, sat: float = 0.0, bri: float = 0.0):
    return _IMAGE_CACHE.get(_cache_key(data, n, sat, bri))


def _cache_set(data: bytes, n: int, sat: float, bri: float, colors, counts, stats):
    key = _cache_key(data, n, sat, bri)
    if len(_IMAGE_CACHE) >= _CACHE_MAX:
        try:
            _IMAGE_CACHE.pop(next(iter(_IMAGE_CACHE)))
        except (KeyError, StopIteration):
            pass
    _IMAGE_CACHE[key] = (colors, counts, stats)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@bot.event
async def on_ready():
    print(f"Ready — logged in as {bot.user} (id: {bot.user.id})")
    if not _post_scheduled_challenges.is_running():
        _post_scheduled_challenges.start()


def _is_guild_admin(member: discord.Member) -> bool:
    p = member.guild_permissions
    return p.administrator or p.manage_guild


@bot.check
async def _require_bot_role(ctx: discord.ApplicationContext) -> bool:
    if not ctx.guild:
        return True
    # Prefer guild cache — has full live role data vs. interaction snapshot
    member = ctx.guild.get_member(ctx.author.id) or ctx.author
    if ctx.guild.owner_id == member.id:
        return True
    if isinstance(member, discord.Member) and _is_guild_admin(member):
        return True
    required_role_id = _get_guild_required_role(ctx.guild_id)
    if not required_role_id:
        return True
    return isinstance(member, discord.Member) and any(
        role.id == int(required_role_id) for role in member.roles
    )


@bot.check
async def _cooldown_check(ctx: discord.ApplicationContext) -> bool:
    """Per-user rate limit on the expensive image commands.

    Runs after _require_bot_role (registered first), so cooldown is only
    checked (not consumed) here.  The timestamp is stamped by
    ``_consume_cooldown`` *after* input validation succeeds, so a rejected
    upload (wrong file type, too large, etc.) does not lock the user out.
    """
    if ctx.command is None or ctx.command.name not in _HEAVY_COMMANDS:
        return True
    now = time.monotonic()
    last = _USER_COOLDOWNS.get(ctx.author.id, 0.0)
    remaining = _COOLDOWN_SECONDS - (now - last)
    if remaining > 0:
        raise _CooldownError(remaining)
    return True


def _consume_cooldown(user_id: int) -> None:
    """Stamp the cooldown so the user must wait before the next heavy command.

    Call this right after input validation succeeds — never before, so that
    rejected requests do not waste the user's cooldown window.

    Opportunistically prunes any entries that have already expired, so
    _USER_COOLDOWNS never accumulates a permanent entry per user (mirrors the
    bounded-size eviction used by _IMAGE_CACHE above).
    """
    now = time.monotonic()
    expired = [
        uid for uid, ts in _USER_COOLDOWNS.items()
        if now - ts >= _COOLDOWN_SECONDS
    ]
    for uid in expired:
        _USER_COOLDOWNS.pop(uid, None)
    _USER_COOLDOWNS[user_id] = now


@bot.event
async def on_application_command_error(ctx: discord.ApplicationContext, error):
    if isinstance(error, _CooldownError):
        await ctx.respond(
            f"You're using commands too quickly — try again in "
            f"{error.retry_after:.0f}s.",
            ephemeral=True,
        )
    elif isinstance(error, commands.CheckFailure):
        required_role_id = _get_guild_required_role(ctx.guild_id) if ctx.guild else None
        role_mention = f"<@&{required_role_id}>" if required_role_id else "the required role"
        await ctx.respond(
            f"You need the {role_mention} role to use this bot.",
            ephemeral=True,
        )
    else:
        raise error


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if not isinstance(message.channel, discord.Thread) or not message.guild:
        return
    channel_id = _get_guild_channel(message.guild.id)
    if not channel_id or message.channel.parent_id != int(channel_id):
        return
    if any(
        a.content_type and a.content_type.startswith("image/")
        for a in message.attachments
    ):
        await message.add_reaction("⭐")


def _pct_bar(pct: float, width: int = 8) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _hue_range_label(hue_range: tuple[int, int]) -> str:
    if hue_range == (None, None):
        return "N/A"
    hue_names = {
        (0, 30): "Red", (30, 60): "Orange", (60, 90): "Yellow",
        (90, 150): "Green", (150, 210): "Cyan", (210, 270): "Blue",
        (270, 330): "Violet", (330, 360): "Red",
    }
    start, end = hue_range
    hi_end = end + 360 if end < start else end
    mid = ((start + hi_end) / 2) % 360
    for (lo, hi), name in hue_names.items():
        if lo <= mid < hi:
            return f"{name} ({start}–{end}°)"
    return f"{start}–{end}°"


def _contrast_level(contrast: float) -> str:
    label = "Low" if contrast < 30 else "Medium" if contrast < 70 else "High"
    return f"{label} ({contrast})"


def _saturation_level(sat_pct: float) -> str:
    label = "Low" if sat_pct < 20 else "Medium" if sat_pct < 60 else "High"
    return f"{label} ({sat_pct}%)"


def _color_line(rgb, cnt, total, show_rgb: bool, show_cmyk: bool) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    hex_str = f"#{r:02X}{g:02X}{b:02X}"
    name = nearest_color_name((r, g, b))
    pct = cnt / total * 100
    bar = _pct_bar(pct)
    line = f"`{hex_str}` **{name}** {bar} {pct:.1f}%"
    if show_rgb:
        line += f"\n  RGB({r}, {g}, {b})"
    if show_cmyk:
        c, m, y, k = rgb_to_cmyk(r, g, b)
        line += f"\n  CMYK({c}%, {m}%, {y}%, {k}%)"
    return line


def _image_rejection_reason(image) -> str | None:
    """Validate an attachment, returning an error string or None if it's OK.

    Shared by all image commands so validation is identical everywhere. The
    pixel-dimension check uses Discord's attachment metadata to reject
    decompression bombs *before* download; load_image_from_bytes enforces the
    same MAX_IMAGE_PIXELS limit again on the decoded raster as defense in depth.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        return f"`{image.filename}` is not a valid image file (PNG, JPEG, etc.)."
    if image.size > MAX_FILE_BYTES:
        return (
            f"`{image.filename}` is too large (max 15 MB). "
            f"It's {image.size / 1e6:.1f} MB."
        )
    if image.width and image.height and image.width * image.height > MAX_IMAGE_PIXELS:
        return (
            f"`{image.filename}` has too many pixels "
            f"({image.width}×{image.height}); max is {MAX_IMAGE_PIXELS:,}."
        )
    return None


async def _validate_image(ctx, image) -> bool:
    reason = _image_rejection_reason(image)
    if reason:
        await ctx.followup.send(reason)
        return False
    return True


def _build_stats_embed(
    stats: dict, colors, counts, image_name: str,
    grayscale_warning: bool, show_rgb: bool = False, show_cmyk: bool = False,
) -> discord.Embed:
    total = counts.sum()
    embed = discord.Embed(
        title=f"Image Analysis — {image_name}",
        color=discord.Color.from_rgb(int(colors[0][0]), int(colors[0][1]), int(colors[0][2])),
    )
    embed.add_field(name="Dimensions",       value=f"{stats['width']} × {stats['height']} px", inline=True)
    embed.add_field(name="Brightness",       value=f"{stats['brightness']} / 255",              inline=True)
    embed.add_field(name="Contrast",         value=_contrast_level(stats["contrast"]),           inline=True)
    embed.add_field(name="Mean Saturation",  value=_saturation_level(stats["mean_saturation_pct"]), inline=True)
    embed.add_field(name="Dominant Hue",     value=_hue_range_label(stats["dominant_hue_range"]), inline=True)
    embed.add_field(name="Palette Type",     value=classify_palette_type(colors, counts),        inline=True)

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
    embed.set_thumbnail(url="attachment://hue_sat.png")
    return embed


def _build_gradient_embed(filename, gradient_label, img, gradient_stops) -> discord.Embed:
    _, r0, g0, b0 = gradient_stops[0]
    _, r1, g1, b1 = gradient_stops[-1]
    mid_r, mid_g, mid_b = (r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2
    embed = discord.Embed(
        title=f"Gradient Map — {filename}",
        color=discord.Color.from_rgb(mid_r, mid_g, mid_b),
    )
    embed.add_field(name="Gradient",   value=gradient_label,                      inline=True)
    embed.add_field(name="Dimensions", value=f"{img.width} × {img.height} px",    inline=True)
    embed.set_image(url="attachment://gradient_map.png")
    embed.set_thumbnail(url="attachment://gradient_swatch.png")
    return embed


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------

@bot.slash_command(name="analyze", description="Full analysis: dominant colors, stats, and charts", guild_ids=guild_ids)
async def analyze(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Upload a painting image"),
    num_colors: discord.Option(int, description="Number of dominant colors to extract (default 10)",
                               default=10, min_value=3, max_value=16),
    show_rgb: discord.Option(bool, description="Show RGB values for each color", default=False, required=False),
    show_cmyk: discord.Option(bool, description="Show CMYK values for each color", default=False, required=False),
    saturation_boost: discord.Option(float, description="Adjust saturation: -1.0 (grayscale) to +1.0 (vivid). Default 0.",
                                     default=0.0, min_value=-1.0, max_value=1.0, required=False),
    brightness_boost: discord.Option(float, description="Adjust brightness: -1.0 (dark) to +1.0 (bright). Default 0.",
                                     default=0.0, min_value=-1.0, max_value=1.0, required=False),
):
    await ctx.defer()
    if not await _validate_image(ctx, image):
        return
    _consume_cooldown(ctx.author.id)
    try:
        data = await image.read()

        def _work():
            cached = _cache_get(data, num_colors, saturation_boost, brightness_boost)
            if cached:
                colors, counts, stats = cached
                img = load_image_from_bytes(data)
                if saturation_boost != 0.0 or brightness_boost != 0.0:
                    img = adjust_image(img, saturation_boost, brightness_boost)
            else:
                img = load_image_from_bytes(data)
                if saturation_boost != 0.0 or brightness_boost != 0.0:
                    img = adjust_image(img, saturation_boost, brightness_boost)
                colors, counts = extract_dominant_colors(img, n=num_colors)
                stats = compute_stats(img)
                _cache_set(data, num_colors, saturation_boost, brightness_boost, colors, counts, stats)
            palette_buf = render_chart_to_bytesio(render_palette_chart(colors, counts))
            hue_sat_buf = render_chart_to_bytesio(render_hue_saturation_chart(img))
            return colors, counts, stats, palette_buf, hue_sat_buf

        colors, counts, stats, palette_buf, hue_sat_buf = await _run_cpu(_work)

        grayscale_warning = stats["mean_saturation_pct"] < 15
        embed = _build_stats_embed(stats, colors, counts, image.filename, grayscale_warning, show_rgb, show_cmyk)

        if saturation_boost != 0.0 or brightness_boost != 0.0:
            adj_note = []
            if saturation_boost != 0.0:
                adj_note.append(f"saturation {saturation_boost:+.1f}")
            if brightness_boost != 0.0:
                adj_note.append(f"brightness {brightness_boost:+.1f}")
            embed.set_footer(text=f"Adjusted: {', '.join(adj_note)}")

        await ctx.followup.send(
            embed=embed,
            files=[
                discord.File(palette_buf,  filename="palette.png"),
                discord.File(hue_sat_buf,  filename="hue_sat.png"),
            ],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong analyzing that image. Make sure it's a valid image file.")


# ---------------------------------------------------------------------------
# /palette
# ---------------------------------------------------------------------------

@bot.slash_command(name="palette", description="Quick color palette swatch only", guild_ids=guild_ids)
async def palette(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Upload a painting image"),
    num_colors: discord.Option(int, description="Number of dominant colors to extract (default 10)",
                               default=10, min_value=3, max_value=16),
    show_rgb: discord.Option(bool, description="Show RGB values for each color", default=False, required=False),
    show_cmyk: discord.Option(bool, description="Show CMYK values for each color", default=False, required=False),
    saturation_boost: discord.Option(float, description="Adjust saturation before extracting (-1 to +1). Default 0.",
                                     default=0.0, min_value=-1.0, max_value=1.0, required=False),
    brightness_boost: discord.Option(float, description="Adjust brightness before extracting (-1 to +1). Default 0.",
                                     default=0.0, min_value=-1.0, max_value=1.0, required=False),
):
    await ctx.defer()
    if not await _validate_image(ctx, image):
        return
    _consume_cooldown(ctx.author.id)
    try:
        data = await image.read()

        def _work():
            cached = _cache_get(data, num_colors, saturation_boost, brightness_boost)
            if cached:
                colors, counts, _ = cached
            else:
                img = load_image_from_bytes(data)
                if saturation_boost != 0.0 or brightness_boost != 0.0:
                    img = adjust_image(img, saturation_boost, brightness_boost)
                colors, counts = extract_dominant_colors(img, n=num_colors)
                stats = compute_stats(img)
                _cache_set(data, num_colors, saturation_boost, brightness_boost, colors, counts, stats)
            palette_buf = render_chart_to_bytesio(render_palette_chart(colors, counts))
            return colors, counts, palette_buf

        colors, counts, palette_buf = await _run_cpu(_work)
        total = counts.sum()
        lines = [_color_line(rgb, cnt, total, show_rgb, show_cmyk) for rgb, cnt in zip(colors, counts)]
        embed = discord.Embed(
            title=f"Color Palette — {image.filename}",
            description="\n".join(lines),
            color=discord.Color.from_rgb(int(colors[0][0]), int(colors[0][1]), int(colors[0][2])),
        )
        embed.set_image(url="attachment://palette.png")

        await ctx.followup.send(
            embed=embed,
            files=[discord.File(palette_buf, filename="palette.png")],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong. Make sure it's a valid image file.")


# ---------------------------------------------------------------------------
# /gradient_map
# ---------------------------------------------------------------------------

@bot.slash_command(name="gradient_map", description="Remap image tones through a color gradient", guild_ids=guild_ids)
async def gradient_map_cmd(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Image to process", required=True),
    preset: discord.Option(str, description="Predefined gradient (default: fire)",
                           choices=["fire", "ocean", "forest", "amethyst", "grayscale", "sunset", "ice"],
                           required=False, default="fire"),
    start_color: discord.Option(str, description="Custom 2-stop: shadow hex (e.g. #1a0030) — pair with end_color",
                                required=False, default=None),
    end_color: discord.Option(str, description="Custom 2-stop: highlight hex (e.g. #ffe080) — pair with start_color",
                              required=False, default=None),
    custom_colors: discord.Option(str,
                                  description="Multi-stop gradient: comma-separated hex list (e.g. #1a0030,#7b2d8b,#ffe080)",
                                  required=False, default=None),
    reverse: discord.Option(bool, description="Flip the gradient direction", default=False, required=False),
):
    await ctx.defer()
    if not await _validate_image(ctx, image):
        return
    _consume_cooldown(ctx.author.id)

    # Resolve gradient stops
    gradient_stops = None
    gradient_label = preset

    if custom_colors is not None:
        try:
            gradient_stops = parse_multi_hex_gradient(custom_colors)
            hex_labels = " → ".join(p.strip().upper() for p in custom_colors.split(",") if p.strip())
            gradient_label = f"custom ({hex_labels})"
        except ValueError as e:
            await ctx.followup.send(str(e))
            return
    elif start_color is not None or end_color is not None:
        if (start_color is None) != (end_color is None):
            await ctx.followup.send("Provide both `start_color` and `end_color`, or neither.")
            return
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

    if reverse:
        gradient_stops = reverse_gradient(gradient_stops)
        gradient_label += " (reversed)"

    try:
        data = await image.read()

        def _work():
            img = load_image_from_bytes(data)
            result = apply_gradient_map(img, gradient_stops)
            out_buf = io.BytesIO()
            result.save(out_buf, format="PNG")
            out_buf.seek(0)
            swatch_buf = render_gradient_preview(gradient_stops)
            return img, out_buf, swatch_buf

        img, out_buf, swatch_buf = await _run_cpu(_work)
        embed = _build_gradient_embed(image.filename, gradient_label, img, gradient_stops)

        await ctx.followup.send(
            embed=embed,
            files=[
                discord.File(out_buf,    filename="gradient_map.png"),
                discord.File(swatch_buf, filename="gradient_swatch.png"),
            ],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong processing the image.")


# ---------------------------------------------------------------------------
# /palette_gradient
# ---------------------------------------------------------------------------

@bot.slash_command(name="palette_gradient",
                   description="Generate a gradient from the image's own colors and apply it as a tone map",
                   guild_ids=guild_ids)
async def palette_gradient_cmd(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Image to process", required=True),
    num_colors: discord.Option(int, description="Number of colors to extract for the gradient (default 5)",
                               default=5, min_value=3, max_value=100),
    sort_by: discord.Option(str, description="How to order colors in the gradient (default: value)",
                            choices=["value", "luminance", "hue", "saturation"], default="value"),
    reverse: discord.Option(bool, description="Flip the gradient direction", default=False, required=False),
):
    await ctx.defer()
    if not await _validate_image(ctx, image):
        return
    _consume_cooldown(ctx.author.id)
    try:
        data = await image.read()

        def _work():
            img = load_image_from_bytes(data)
            colors, counts = extract_dominant_colors(img, n=num_colors)
            gradient_stops = palette_to_gradient_stops(colors, counts, sort_by=sort_by)
            if reverse:
                gradient_stops = reverse_gradient(gradient_stops)
            result = apply_gradient_map(img, gradient_stops)
            out_buf = io.BytesIO()
            result.save(out_buf, format="PNG")
            out_buf.seek(0)
            swatch_buf = render_gradient_preview(gradient_stops)
            return colors, gradient_stops, img, out_buf, swatch_buf

        colors, gradient_stops, img, out_buf, swatch_buf = await _run_cpu(_work)
        stop_colors = " → ".join(f"#{r:02X}{g:02X}{b:02X}" for _, r, g, b in sorted(gradient_stops))
        embed = discord.Embed(
            title=f"Palette Gradient — {image.filename}",
            color=discord.Color.from_rgb(*[int(v) for v in colors[0]]),
        )
        embed.add_field(name="Gradient",    value=stop_colors,                       inline=False)
        embed.add_field(name="Sorted by",   value=sort_by.capitalize(),              inline=True)
        embed.add_field(name="Reversed",    value="Yes" if reverse else "No",        inline=True)
        embed.add_field(name="Dimensions",  value=f"{img.width} × {img.height} px", inline=True)
        embed.set_image(url="attachment://gradient_map.png")
        embed.set_thumbnail(url="attachment://gradient_swatch.png")

        await ctx.followup.send(
            embed=embed,
            files=[
                discord.File(out_buf,    filename="gradient_map.png"),
                discord.File(swatch_buf, filename="gradient_swatch.png"),
            ],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong processing the image.")


# ---------------------------------------------------------------------------
# /export_palette
# ---------------------------------------------------------------------------

def _sanitize_filename_component(name: str) -> str:
    """Sanitize a user-supplied string for safe use as a filename component.

    Strips control characters (including \\r, \\n) and replaces path
    separators and other filesystem-unsafe characters with underscores,
    so a malicious ``palette_name`` can't inject newlines or traverse
    directories via the attachment filename.
    """
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = name.strip().strip(".")
    return name or "Palette"


@bot.slash_command(name="export_palette",
                   description="Export dominant colors as a palette file for design software",
                   guild_ids=guild_ids)
async def export_palette_cmd(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Image to extract colors from", required=True),
    format: discord.Option(str, description="Export format",
                           choices=["ase", "swatches", "gpl", "aco", "css", "tailwind"], default="ase"),
    num_colors: discord.Option(int, description="Number of colors to extract (default 10)",
                               default=10, min_value=3, max_value=16),
    palette_name: discord.Option(str, description="Name for the palette (default: Palette)",
                                 default="Palette", required=False),
):
    await ctx.defer()
    if not await _validate_image(ctx, image):
        return
    _consume_cooldown(ctx.author.id)
    try:
        data = await image.read()

        def _work():
            img = load_image_from_bytes(data)
            colors, counts = extract_dominant_colors(img, n=num_colors)
            color_list = [(int(c[0]), int(c[1]), int(c[2])) for c in colors]
            safe_palette_name = _sanitize_filename_component(palette_name)
            format_map = {
                "ase":      (export_ase,      "palette.ase"),
                "swatches": (export_swatches, "palette.swatches"),
                "gpl":      (export_gpl,      f"{safe_palette_name}.gpl"),
                "aco":      (export_aco,      "palette.aco"),
                "css":      (export_css,      "palette.css"),
                "tailwind": (export_tailwind, "palette.json"),
            }
            fn, filename = format_map[format]
            file_bytes = fn(color_list, palette_name)
            return colors, counts, color_list, file_bytes, filename

        colors, counts, color_list, file_bytes, filename = await _run_cpu(_work)

        total = counts.sum()
        lines = [
            f"`#{r:02X}{g:02X}{b:02X}` {_pct_bar(cnt/total*100)} **{nearest_color_name((r,g,b))}** {cnt/total*100:.1f}%"
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


# ---------------------------------------------------------------------------
# /export_gradient
# ---------------------------------------------------------------------------

@bot.slash_command(name="export_gradient",
                   description="Export a palette-derived gradient as a .ggr (GIMP/Krita) or .json file",
                   guild_ids=guild_ids)
async def export_gradient_cmd(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Image to extract colors from", required=True),
    format: discord.Option(str, description="Export format (default: ggr)",
                           choices=["ggr", "json"], default="ggr"),
    num_colors: discord.Option(int, description="Number of colors to extract (default 5)",
                               default=5, min_value=3, max_value=100),
    sort_by: discord.Option(str, description="How to order colors in the gradient (default: value)",
                            choices=["value", "luminance", "hue", "saturation"], default="value"),
    gradient_name: discord.Option(str, description="Name embedded in the gradient file",
                                  default="palette_gradient", required=False),
    reverse: discord.Option(bool, description="Flip the gradient direction", default=False, required=False),
):
    await ctx.defer()
    if not await _validate_image(ctx, image):
        return
    _consume_cooldown(ctx.author.id)
    try:
        data = await image.read()

        def _work():
            img = load_image_from_bytes(data)
            colors, counts = extract_dominant_colors(img, n=num_colors)
            gradient_stops = palette_to_gradient_stops(colors, counts, sort_by=sort_by)
            if reverse:
                gradient_stops = reverse_gradient(gradient_stops)
            swatch_buf = render_gradient_preview(gradient_stops)
            if format == "ggr":
                file_bytes = export_gradient_ggr(gradient_stops, name=gradient_name)
                filename = f"{gradient_name}.ggr"
            else:
                file_bytes = export_gradient_json(gradient_stops, name=gradient_name)
                filename = f"{gradient_name}.json"
            return colors, gradient_stops, swatch_buf, file_bytes, filename

        colors, gradient_stops, swatch_buf, file_bytes, filename = await _run_cpu(_work)
        stop_colors = " → ".join(f"#{r:02X}{g:02X}{b:02X}" for _, r, g, b in sorted(gradient_stops))
        embed = discord.Embed(
            title=f"Gradient Export — {image.filename}",
            color=discord.Color.from_rgb(*[int(v) for v in colors[0]]),
        )
        embed.add_field(name="Gradient",   value=stop_colors,              inline=False)
        embed.add_field(name="Format",     value=f".{format}",             inline=True)
        embed.add_field(name="Sorted by",  value=sort_by.capitalize(),     inline=True)
        embed.add_field(name="Colors",     value=str(num_colors),          inline=True)
        embed.set_image(url="attachment://gradient_swatch.png")

        await ctx.followup.send(
            embed=embed,
            files=[
                discord.File(io.BytesIO(file_bytes), filename=filename),
                discord.File(swatch_buf, filename="gradient_swatch.png"),
            ],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong exporting the gradient.")


# ---------------------------------------------------------------------------
# /color_info  — NEW
# ---------------------------------------------------------------------------

@bot.slash_command(name="color_info",
                   description="Look up a hex color: RGB, CMYK, HSV, name, and harmony suggestions",
                   guild_ids=guild_ids)
async def color_info_cmd(
    ctx: discord.ApplicationContext,
    hex_color: discord.Option(str, description="Hex color code, e.g. #3a7bd5 or 3a7bd5"),
):
    await ctx.defer()
    try:
        r, g, b = parse_hex_color(hex_color)
    except ValueError as e:
        await ctx.followup.send(str(e))
        return
    _consume_cooldown(ctx.author.id)

    try:
        hex_str = f"#{r:02X}{g:02X}{b:02X}"
        name = nearest_color_name((r, g, b))
        c, m, y, k = rgb_to_cmyk(r, g, b)
        h_f, s_f, v_f = __import__("colorsys").rgb_to_hsv(r / 255, g / 255, b / 255)
        h_deg = round(h_f * 360)
        s_pct = round(s_f * 100)
        v_pct = round(v_f * 100)
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        brightness_label = "Dark" if lum < 64 else "Medium-Dark" if lum < 128 else "Medium-Light" if lum < 192 else "Light"
        # Temperature: warm = reds/oranges/yellows (h < 60° or h > 300°), cool = blues/greens
        warm = h_deg < 60 or h_deg > 300
        temp_label = "Warm" if warm else ("Neutral" if 60 <= h_deg <= 80 or 160 <= h_deg <= 200 else "Cool")

        # Harmony suggestions
        import colorsys as _cs

        def _hue_rgb(target_h):
            rr, gg, bb = _cs.hsv_to_rgb(target_h % 1.0, max(s_f, 0.65), max(v_f, 0.6))
            return (int(rr * 255), int(gg * 255), int(bb * 255))

        harmonies = [
            (_hue_rgb(h_f + 0.5),    "Complement"),
            (_hue_rgb(h_f + 1 / 3),  "Triadic A"),
            (_hue_rgb(h_f + 2 / 3),  "Triadic B"),
            (_hue_rgb(h_f + 1 / 12), "Analogous +30°"),
            (_hue_rgb(h_f - 1 / 12), "Analogous -30°"),
        ]

        swatch_buf = await _run_cpu(render_color_info_swatch, (r, g, b), harmonies)

        embed = discord.Embed(
            title=f"Color Info — {hex_str}",
            color=discord.Color.from_rgb(r, g, b),
        )
        embed.add_field(name="Name",        value=name,                                    inline=True)
        embed.add_field(name="Brightness",  value=brightness_label,                        inline=True)
        embed.add_field(name="Temperature", value=temp_label,                              inline=True)
        embed.add_field(name="RGB",         value=f"R {r}, G {g}, B {b}",                 inline=True)
        embed.add_field(name="HSV",         value=f"H {h_deg}°, S {s_pct}%, V {v_pct}%", inline=True)
        embed.add_field(name="CMYK",        value=f"C {c}%, M {m}%, Y {y}%, K {k}%",     inline=True)
        harmony_text = "\n".join(
            f"`#{rh:02X}{gh:02X}{bh:02X}` {lbl}" for (rh, gh, bh), lbl in harmonies
        )
        embed.add_field(name="Harmony Suggestions", value=harmony_text, inline=False)
        embed.set_image(url="attachment://color_info.png")

        await ctx.followup.send(
            embed=embed,
            files=[discord.File(swatch_buf, filename="color_info.png")],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong processing that color.")


# ---------------------------------------------------------------------------
# /compare  — NEW
# ---------------------------------------------------------------------------

@bot.slash_command(name="compare",
                   description="Compare the dominant palettes of two images side by side",
                   guild_ids=guild_ids)
async def compare_cmd(
    ctx: discord.ApplicationContext,
    image_a: discord.Option(discord.Attachment, description="First image",  required=True),
    image_b: discord.Option(discord.Attachment, description="Second image", required=True),
    num_colors: discord.Option(int, description="Colors to extract per image (default 8)",
                               default=8, min_value=3, max_value=16),
):
    await ctx.defer()
    for att in (image_a, image_b):
        reason = _image_rejection_reason(att)
        if reason:
            await ctx.followup.send(reason)
            return
    _consume_cooldown(ctx.author.id)

    try:
        data_a = await image_a.read()
        data_b = await image_b.read()

        def _work():
            img_a = load_image_from_bytes(data_a)
            img_b = load_image_from_bytes(data_b)
            colors_a, counts_a = extract_dominant_colors(img_a, n=num_colors)
            colors_b, counts_b = extract_dominant_colors(img_b, n=num_colors)
            compare_buf = render_compare_chart(
                colors_a, counts_a, image_a.filename,
                colors_b, counts_b, image_b.filename,
            )
            return colors_a, counts_a, colors_b, counts_b, compare_buf

        colors_a, counts_a, colors_b, counts_b, compare_buf = await _run_cpu(_work)
        total_a, total_b = counts_a.sum(), counts_b.sum()

        def _top_hex(colors, counts, total):
            return "  ".join(
                f"`#{int(c[0]):02X}{int(c[1]):02X}{int(c[2]):02X}` {cnt/total*100:.0f}%"
                for c, cnt in zip(colors[:4], counts[:4])
            )

        embed = discord.Embed(
            title="Palette Comparison",
            color=discord.Color.from_rgb(int(colors_a[0][0]), int(colors_a[0][1]), int(colors_a[0][2])),
        )
        embed.add_field(name=f"🖼 {image_a.filename}", value=_top_hex(colors_a, counts_a, total_a), inline=False)
        embed.add_field(name=f"🖼 {image_b.filename}", value=_top_hex(colors_b, counts_b, total_b), inline=False)
        embed.add_field(name="Palette Type A", value=classify_palette_type(colors_a, counts_a), inline=True)
        embed.add_field(name="Palette Type B", value=classify_palette_type(colors_b, counts_b), inline=True)
        embed.set_image(url="attachment://compare.png")

        await ctx.followup.send(
            embed=embed,
            files=[discord.File(compare_buf, filename="compare.png")],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong comparing the images.")


# ---------------------------------------------------------------------------
# /colorblind  — NEW
# ---------------------------------------------------------------------------

@bot.slash_command(name="colorblind",
                   description="Simulate how your image looks to people with color blindness",
                   guild_ids=guild_ids)
async def colorblind_cmd(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Image to simulate", required=True),
    type: discord.Option(str, description="Type of color blindness to simulate (default: all)",
                         choices=["all", "deuteranopia", "protanopia", "tritanopia"],
                         default="all", required=False),
):
    await ctx.defer()
    if not await _validate_image(ctx, image):
        return
    _consume_cooldown(ctx.author.id)
    try:
        data = await image.read()

        type_labels = {
            "deuteranopia": "Deuteranopia (red-green, missing green)",
            "protanopia":   "Protanopia (red-green, missing red)",
            "tritanopia":   "Tritanopia (blue-yellow, missing blue)",
        }

        def _work():
            img = load_image_from_bytes(data)
            if type == "all":
                return img, render_colorblind_comparison(img)
            sim = simulate_colorblindness(img, type)
            buf = io.BytesIO()
            sim.save(buf, format="PNG")
            buf.seek(0)
            return img, buf

        img, result_buf = await _run_cpu(_work)

        if type == "all":
            filename = "colorblind_all.png"
            type_label = "All types (4-panel comparison)"
        else:
            filename = f"colorblind_{type}.png"
            type_label = type_labels[type]

        embed = discord.Embed(
            title=f"Color Blindness Simulation — {image.filename}",
            color=discord.Color.from_rgb(100, 149, 237),  # cornflower blue — neutral
        )
        embed.add_field(name="Type",       value=type_label,                          inline=False)
        embed.add_field(name="Dimensions", value=f"{img.width} × {img.height} px",   inline=True)
        embed.set_image(url=f"attachment://{filename}")

        await ctx.followup.send(
            embed=embed,
            files=[discord.File(result_buf, filename=filename)],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong simulating color blindness.")


# ---------------------------------------------------------------------------
# /recolor  — NEW
# ---------------------------------------------------------------------------

@bot.slash_command(name="recolor",
                   description="Apply the color palette from one image onto another image",
                   guild_ids=guild_ids)
async def recolor_cmd(
    ctx: discord.ApplicationContext,
    source: discord.Option(discord.Attachment, description="Source image — palette is taken from here", required=True),
    target: discord.Option(discord.Attachment, description="Target image — this gets recolored",        required=True),
    num_colors: discord.Option(int, description="Colors to extract from source (default 8)",
                               default=8, min_value=3, max_value=16),
):
    await ctx.defer()
    for att in (source, target):
        reason = _image_rejection_reason(att)
        if reason:
            await ctx.followup.send(reason)
            return
    _consume_cooldown(ctx.author.id)

    try:
        src_data = await source.read()
        tgt_data = await target.read()

        def _work():
            src_img = load_image_from_bytes(src_data)
            tgt_img = load_image_from_bytes(tgt_data)
            colors, _counts = extract_dominant_colors(src_img, n=num_colors)
            color_list = [(int(c[0]), int(c[1]), int(c[2])) for c in colors]
            result = recolor_image(tgt_img, color_list)
            out_buf = io.BytesIO()
            result.save(out_buf, format="PNG")
            out_buf.seek(0)
            return color_list, out_buf, result.width, result.height

        color_list, out_buf, result_w, result_h = await _run_cpu(_work)

        palette_lines = "  ".join(
            f"`#{r:02X}{g:02X}{b:02X}`" for r, g, b in color_list[:8]
        )

        embed = discord.Embed(
            title=f"Recolor — {target.filename}",
            color=discord.Color.from_rgb(*color_list[0]),
        )
        embed.add_field(name="Source Palette",
                        value=f"From **{source.filename}** ({num_colors} colors)\n{palette_lines}",
                        inline=False)
        embed.add_field(name="Output Size",
                        value=f"{result_w} × {result_h} px", inline=True)
        embed.set_image(url="attachment://recolor.png")

        await ctx.followup.send(
            embed=embed,
            files=[discord.File(out_buf, filename="recolor.png")],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong recoloring the image.")


# ---------------------------------------------------------------------------
# /suggest_harmony  — NEW
# ---------------------------------------------------------------------------

@bot.slash_command(name="suggest_harmony",
                   description="Suggest colors that would harmonize with your image's existing palette",
                   guild_ids=guild_ids)
async def suggest_harmony_cmd(
    ctx: discord.ApplicationContext,
    image: discord.Option(discord.Attachment, description="Upload your painting", required=True),
    num_colors: discord.Option(int, description="Colors to extract from the image (default 8)",
                               default=8, min_value=3, max_value=16),
):
    await ctx.defer()
    if not await _validate_image(ctx, image):
        return
    _consume_cooldown(ctx.author.id)
    try:
        data = await image.read()

        def _work():
            img = load_image_from_bytes(data)
            colors, counts = extract_dominant_colors(img, n=num_colors)
            palette_type, suggestions = suggest_harmony_colors(colors, counts)
            if not suggestions:
                return palette_type, suggestions, None, None
            orig_list = [tuple(int(v) for v in c) for c in colors]
            harmony_buf = render_harmony_chart(orig_list, suggestions)
            return palette_type, suggestions, orig_list, harmony_buf

        palette_type, suggestions, orig_list, harmony_buf = await _run_cpu(_work)

        if not suggestions:
            await ctx.followup.send(
                "Could not determine harmony suggestions — the palette may be mostly grayscale."
            )
            return

        sugg_text = "\n".join(
            f"`#{r:02X}{g:02X}{b:02X}` **{nearest_color_name((r,g,b))}** — {lbl}"
            for (r, g, b), lbl in suggestions
        )

        embed = discord.Embed(
            title=f"Harmony Suggestions — {image.filename}",
            color=discord.Color.from_rgb(*orig_list[0]),
        )
        embed.add_field(name="Detected Palette Type", value=palette_type, inline=True)
        embed.add_field(name="Suggested Colors", value=sugg_text, inline=False)
        embed.set_image(url="attachment://harmony.png")

        await ctx.followup.send(
            embed=embed,
            files=[discord.File(harmony_buf, filename="harmony.png")],
        )
    except Exception:
        traceback.print_exc()
        await ctx.followup.send("Something went wrong generating harmony suggestions.")


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

COMMAND_DOCS = {
    "analyze": {
        "summary": "Full analysis: dominant colors, image stats, and two charts",
        "description": (
            "Extracts dominant colors via KMeans, computes brightness/contrast/saturation/hue stats, "
            "classifies the palette type, and generates a color swatch chart and a hue/saturation "
            "distribution chart. Optional saturation and brightness boosts are applied before extraction."
        ),
        "params": [
            ("`image`",             "required",              "The painting to analyze (PNG, JPEG, max 15 MB)"),
            ("`num_colors`",        "3–16, default 10",      "How many dominant colors to extract"),
            ("`show_rgb`",          "true/false, default false", "Include RGB values alongside each color"),
            ("`show_cmyk`",         "true/false, default false", "Include CMYK values alongside each color"),
            ("`saturation_boost`",  "-1.0 to +1.0, default 0", "Adjust saturation before extracting"),
            ("`brightness_boost`",  "-1.0 to +1.0, default 0", "Adjust brightness before extracting"),
        ],
        "output": "Embed with stats + top 5 colors, `palette.png` swatch chart, `hue_sat.png` distribution chart",
    },
    "palette": {
        "summary": "Quick color palette swatch — no stats, just colors",
        "description": (
            "Extracts dominant colors and lists all of them with hex codes, color names, "
            "a percentage bar, and optional RGB/CMYK values. Faster than /analyze."
        ),
        "params": [
            ("`image`",            "required",                  "The painting (PNG, JPEG, max 15 MB)"),
            ("`num_colors`",       "3–16, default 10",          "How many dominant colors to extract"),
            ("`show_rgb`",         "true/false, default false", "Include RGB values"),
            ("`show_cmyk`",        "true/false, default false", "Include CMYK values"),
            ("`saturation_boost`", "-1.0 to +1.0, default 0",  "Adjust saturation before extracting"),
            ("`brightness_boost`", "-1.0 to +1.0, default 0",  "Adjust brightness before extracting"),
        ],
        "output": "Embed listing all colors with hex/name/bar/%, attached `palette.png`",
    },
    "gradient_map": {
        "summary": "Remap image tones through a color gradient",
        "description": (
            "Applies a gradient map — each pixel's luminance is remapped to a color from the chosen "
            "gradient, preserving light/dark structure. Use a preset, a 2-stop custom pair, "
            "or a comma-separated multi-stop hex list. Add `reverse:True` to flip the gradient."
        ),
        "params": [
            ("`image`",         "required",          "Image to process (max 15 MB)"),
            ("`preset`",        "fire/ocean/forest/amethyst/grayscale/sunset/ice, default fire", "Built-in gradient"),
            ("`start_color`",   "hex, optional",     "Custom 2-stop shadow color — pair with end_color"),
            ("`end_color`",     "hex, optional",     "Custom 2-stop highlight color — pair with start_color"),
            ("`custom_colors`", "hex list, optional","Multi-stop: `#1a0030,#7b2d8b,#ffe080` overrides preset"),
            ("`reverse`",       "true/false",        "Flip the gradient direction"),
        ],
        "output": "Embed with label and dimensions, `gradient_map.png` result, `gradient_swatch.png` preview",
    },
    "palette_gradient": {
        "summary": "Auto-generate a gradient from the image's own colors and apply it",
        "description": (
            "Extracts dominant colors, sorts them by the chosen dimension, applies as a tone map. "
            "Use `reverse:True` to flip. Export the resulting gradient with /export_gradient."
        ),
        "params": [
            ("`image`",       "required",                                   "Image to process (max 15 MB)"),
            ("`num_colors`",  "3–100, default 5",                           "Colors to extract for the gradient"),
            ("`sort_by`",     "value/luminance/hue/saturation, default value", "How to order colors in the gradient"),
            ("`reverse`",     "true/false",                                 "Flip the gradient direction"),
        ],
        "output": "Embed with gradient stops; `gradient_map.png` result, `gradient_swatch.png` preview",
    },
    "export_palette": {
        "summary": "Export colors as .ase, .swatches, .gpl, .aco, .css, or .json (Tailwind)",
        "description": (
            "Extracts dominant colors and exports a palette file. "
            "`.ase` = Adobe (Photoshop/Illustrator/InDesign). "
            "`.swatches` = Procreate. "
            "`.gpl` = GIMP Palette. "
            "`.aco` = Photoshop Color Swatch. "
            "`.css` = CSS custom properties. "
            "`.json` = Tailwind config color extension."
        ),
        "params": [
            ("`image`",        "required",                           "Image to extract colors from (max 15 MB)"),
            ("`format`",       "ase/swatches/gpl/aco/css/tailwind",  "Export format"),
            ("`num_colors`",   "3–16, default 10",                   "Number of colors to extract"),
            ("`palette_name`", "text, default `Palette`",            "Name embedded in the file"),
        ],
        "output": "Embed listing all colors, attached palette file for download",
    },
    "export_gradient": {
        "summary": "Export a palette-derived gradient as a .ggr (GIMP/Krita) or .json file",
        "description": (
            "Extracts dominant colors, sorts into gradient stops, and exports. "
            "`.ggr` = GIMP/Krita. `.json` = generic. Use `reverse:True` to flip."
        ),
        "params": [
            ("`image`",         "required",                              "Image (max 15 MB)"),
            ("`format`",        "ggr/json, default ggr",                "Export format"),
            ("`num_colors`",    "3–100, default 5",                     "Colors to extract"),
            ("`sort_by`",       "value/luminance/hue/saturation",        "Sorting dimension"),
            ("`gradient_name`", "text, default `palette_gradient`",      "Name in the file"),
            ("`reverse`",       "true/false",                            "Flip the gradient direction"),
        ],
        "output": "Embed with gradient stops, attached gradient file",
    },
    "color_info": {
        "summary": "Look up any hex color: RGB, CMYK, HSV, name, and harmony suggestions",
        "description": (
            "No image needed. Provide any hex color and get back all color representations, "
            "a nearest named-color match, perceived brightness, temperature (warm/cool), "
            "and a set of harmony suggestions (complement, triadic, analogous) with a swatch chart."
        ),
        "params": [
            ("`hex_color`", "required", "Hex color code, e.g. `#3a7bd5` or `3a7bd5`"),
        ],
        "output": "Embed with color data + harmony list, `color_info.png` swatch",
    },
    "compare": {
        "summary": "Side-by-side palette comparison of two images",
        "description": (
            "Uploads two images and extracts a dominant palette from each. "
            "Renders a two-row chart showing both palettes proportionally, "
            "and reports the palette type of each."
        ),
        "params": [
            ("`image_a`",    "required",         "First image (max 15 MB)"),
            ("`image_b`",    "required",         "Second image (max 15 MB)"),
            ("`num_colors`", "3–16, default 8",  "Colors to extract per image"),
        ],
        "output": "Embed with top colors from each image, `compare.png` two-row chart",
    },
    "colorblind": {
        "summary": "Simulate how your image looks with color blindness",
        "description": (
            "Renders the image as it appears to viewers with deuteranopia (red-green, missing green), "
            "protanopia (red-green, missing red), or tritanopia (blue-yellow). "
            "Choose `all` (default) for a 4-panel comparison grid, or pick a specific type."
        ),
        "params": [
            ("`image`", "required", "Image to simulate (max 15 MB)"),
            ("`type`",  "all/deuteranopia/protanopia/tritanopia, default all", "Which simulation to run"),
        ],
        "output": "4-panel comparison image (all) or single simulation PNG",
    },
    "recolor": {
        "summary": "Apply one image's color palette onto another image",
        "description": (
            "Extracts dominant colors from the source image, then remaps every pixel in the target "
            "image to its nearest color in that palette. Creates a stylized 'palette transfer' effect "
            "popular in concept art workflows."
        ),
        "params": [
            ("`source`",     "required",        "Image whose palette is used (max 15 MB)"),
            ("`target`",     "required",        "Image to recolor (max 15 MB)"),
            ("`num_colors`", "3–16, default 8", "Colors to extract from source"),
        ],
        "output": "Embed showing the source palette used, `recolor.png` result",
    },
    "suggest_harmony": {
        "summary": "Suggest colors that would harmonize with your image's palette",
        "description": (
            "Analyzes the palette type of your image and suggests 2–3 colors that would "
            "harmonize well but are not already present. Monochromatic/Analogous palettes get "
            "complement and split-complement suggestions; Complementary palettes get triadic additions; etc."
        ),
        "params": [
            ("`image`",      "required",         "Upload your painting (max 15 MB)"),
            ("`num_colors`", "3–16, default 8",  "Colors to extract from the image"),
        ],
        "output": "Embed with suggested hex colors and labels, `harmony.png` swatch chart",
    },
}

_HELP_CHOICES = list(COMMAND_DOCS.keys())


@bot.slash_command(name="help", description="Show available commands and how to use them", guild_ids=guild_ids)
async def help_cmd(
    ctx: discord.ApplicationContext,
    command: discord.Option(str, description="Get detailed help for a specific command",
                            choices=_HELP_CHOICES, required=False, default=None),
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
        embed.add_field(name="Output",     value=doc["output"], inline=False)
        await ctx.respond(embed=embed)


# ---------------------------------------------------------------------------
# Daily challenge — helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Persistence
#
# Single-process state. Config is cached in memory (so the hot path — e.g.
# on_message — never touches disk), and every write is atomic (temp file +
# os.replace) and serialized behind an asyncio.Lock so concurrent handlers and
# the background loop can't clobber each other's writes. (A SQLite/aiosqlite
# datastore is the next step if the bot is ever run as multiple instances.)
# ---------------------------------------------------------------------------

_CONFIG_LOCK = asyncio.Lock()
_SCHEDULE_LOCK = asyncio.Lock()
_config_cache: dict | None = None


def _atomic_write_json(path: str, data) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)  # atomic on POSIX and Windows


def _config() -> dict:
    """Return the in-memory config, loading it from disk once on first use."""
    global _config_cache
    if _config_cache is None:
        try:
            with open(_CONFIG_FILE, encoding="utf-8") as f:
                loaded = json.load(f)
            _config_cache = loaded if isinstance(loaded, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            _config_cache = {}
    return _config_cache


async def _persist_config() -> None:
    async with _CONFIG_LOCK:
        await asyncio.to_thread(_atomic_write_json, _CONFIG_FILE, _config())


def _load_references() -> list[str]:
    try:
        with open(_REFERENCES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_schedule() -> list[dict]:
    try:
        with open(_SCHEDULE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_schedule(challenges: list[dict]) -> None:
    _atomic_write_json(_SCHEDULE_FILE, challenges)


def _get_guild_channel(guild_id: int) -> str | None:
    return _config().get("guild_channels", {}).get(str(guild_id))


async def _set_guild_channel(guild_id: int, channel_id: str) -> None:
    _config().setdefault("guild_channels", {})[str(guild_id)] = channel_id
    await _persist_config()


def _get_guild_required_role(guild_id: int) -> str | None:
    return _config().get("guild_required_roles", {}).get(str(guild_id))


async def _set_guild_required_role(guild_id: int, role_id: str) -> None:
    _config().setdefault("guild_required_roles", {})[str(guild_id)] = role_id
    await _persist_config()


def _get_guild_daily_role(guild_id: int) -> str | None:
    return _config().get("guild_daily_roles", {}).get(str(guild_id))


async def _set_guild_daily_role(guild_id: int, role_id: str) -> None:
    _config().setdefault("guild_daily_roles", {})[str(guild_id)] = role_id
    await _persist_config()


_TIME_RE = re.compile(
    r"^(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ampm>am|pm)?$",
    re.IGNORECASE,
)

_DATE_YMD_RE   = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATE_MDY_RE   = re.compile(r"^(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?$")
_DATE_MONTH_RE = re.compile(
    r"^(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+(\d{1,2})$",
    re.IGNORECASE,
)
_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_release_datetime(time_str: str, date_str: str | None = None) -> str:
    """Return an ISO-8601 datetime string in ET.

    time_str: time of day, e.g. "18:00", "6pm", "6:30pm"
    date_str: optional date; if None, uses today or tomorrow when time has passed.
              Supported: "today", "tomorrow", "YYYY-MM-DD", "MM/DD", "MM/DD/YYYY",
              "Month D", "Month DD" (e.g. "June 20").
    """
    m = _TIME_RE.match(time_str.strip())
    if not m:
        raise ValueError(f"Cannot parse time: {time_str!r}")
    h = int(m.group("h"))
    mins = int(m.group("m") or 0)
    ampm = (m.group("ampm") or "").lower()
    if ampm == "pm" and h != 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0

    now_et = datetime.now(ET)

    if date_str is None:
        naive = datetime(now_et.year, now_et.month, now_et.day, h, mins, 0)
        target = ET.localize(naive)
        if target <= now_et:
            target = ET.normalize(target + timedelta(days=1))
        return target.isoformat()

    date_str = date_str.strip().lower()
    if date_str == "today":
        base = now_et.date()
    elif date_str == "tomorrow":
        base = (now_et + timedelta(days=1)).date()
    elif m2 := _DATE_YMD_RE.match(date_str):
        base = _date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
    elif m2 := _DATE_MDY_RE.match(date_str):
        month, day_n = int(m2.group(1)), int(m2.group(2))
        year_raw = m2.group(3)
        year = int(year_raw) if year_raw else now_et.year
        if year < 100:
            year += 2000
        base = _date(year, month, day_n)
        if not year_raw and base < now_et.date():
            base = _date(year + 1, month, day_n)
    elif m2 := _DATE_MONTH_RE.match(date_str):
        month = _MONTH_NAMES[m2.group(1)[:3].lower()]
        day_n = int(m2.group(2))
        year = now_et.year
        candidate = _date(year, month, day_n)
        if candidate < now_et.date():
            candidate = _date(year + 1, month, day_n)
        base = candidate
    else:
        raise ValueError(f"Cannot parse date: {date_str!r}")

    target = ET.localize(datetime(base.year, base.month, base.day, h, mins, 0))
    return target.isoformat()


def _random_minimum_time() -> str:
    minutes = random.randint(1, 15)
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


DISCORD_CONTENT_LIMIT = 2000


def _format_daily_post(challenge: dict) -> str:
    reference = challenge.get("reference")
    min_time  = challenge.get("minimum_time", "")
    extra     = challenge.get("extra_challenge")
    guild_id  = challenge.get("guild_id")

    daily_role_id = _get_guild_daily_role(int(guild_id)) if guild_id else None

    lines = ["DAILY GESTURE"]
    if daily_role_id:
        lines.append(f"<@&{daily_role_id}>")

    lines += ["", "□  □  □"]

    if reference:
        lines += ["", "REFERENCE", reference]

    lines += ["", "MINIMUM TIME", min_time]

    if extra:
        lines += ["", "EXTRA CHALLENGE", extra]

    lines += ["", "□  □  □"]
    return "\n".join(lines)


async def _send_daily_challenge(challenge: dict) -> bool:
    """Post a single scheduled challenge. Returns True on success, False on failure."""
    guild_id = challenge.get("guild_id")
    if not guild_id:
        print("Daily challenge: missing guild_id, skipping.")
        return False
    channel_id = challenge.get("channel_id") or _get_guild_channel(int(guild_id))
    if not channel_id:
        print(f"Daily challenge: no channel configured for guild {guild_id}, skipping.")
        return False
    channel = bot.get_channel(int(channel_id))
    if channel is None:
        print(f"Daily challenge: channel {channel_id} not found for guild {guild_id}.")
        return False
    content = _format_daily_post(challenge)
    day = challenge.get("day", "")
    thread_name = f"[ DAILY GESTURE ] — {day}" if day else "[ DAILY GESTURE ]"
    daily_role_id = _get_guild_daily_role(int(guild_id))
    allowed = discord.AllowedMentions(
        everyone=False,
        users=False,
        roles=[discord.Object(int(daily_role_id))] if daily_role_id else False,
    )
    try:
        if isinstance(channel, discord.ForumChannel):
            await channel.create_thread(name=thread_name, content=content, allowed_mentions=allowed)
        else:
            await channel.send(content=content, allowed_mentions=allowed)
        return True
    except Exception:
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Daily challenge — background task
# ---------------------------------------------------------------------------

@tasks.loop(minutes=1)
async def _post_scheduled_challenges() -> None:
    """Tick function: deliver any challenges whose post_at time has arrived.

    - Malformed/missing post_at entries are logged and dropped.
    - Failed deliveries are kept in the schedule for retry.
    - Entries more than _CHALLENGE_EXPIRY_HOURS past their post_at are dropped
      so permanently-undeliverable entries do not accumulate indefinitely.

    The lock is held during disk I/O but released for network sends so a
    concurrent /daily_challenge can append without waiting on Discord API.
    """
    now = datetime.now(ET)

    # --- Phase 1: classify entries under the lock ---
    async with _SCHEDULE_LOCK:
        schedule = await asyncio.to_thread(_load_schedule)
        due: list[tuple[dict, datetime]] = []
        remaining: list[dict] = []

        for challenge in schedule:
            try:
                post_at = datetime.fromisoformat(challenge["post_at"])
            except (KeyError, ValueError, TypeError):
                print(f"Daily challenge: bad post_at, dropping entry: {challenge!r}")
                continue

            if now >= post_at:
                due.append((challenge, post_at))
            else:
                remaining.append(challenge)

    # --- Phase 2: send due entries outside the lock (network-bound) ---
    # Collect the IDs of all entries seen in Phase 1 so we can distinguish
    # newly-added entries (by /daily_challenge) from stale ones in Phase 3.
    phase1_ids: set[str] = set()
    for c in schedule:
        cid = c.get("id")
        if cid:
            phase1_ids.add(cid)

    # IDs of challenges that failed delivery and should be retried.
    # For entries without an ID, collect them in failed_noID for fallback.
    failed_ids: set[str] = set()
    failed_noID: list[dict] = []
    for challenge, post_at in due:
        age = now - post_at
        if age > timedelta(hours=_CHALLENGE_EXPIRY_HOURS):
            print(
                f"Daily challenge: expired (>{_CHALLENGE_EXPIRY_HOURS}h overdue), "
                f"dropping: {challenge!r}"
            )
            continue

        ok = await _send_daily_challenge(challenge)
        if not ok:
            cid = challenge.get("id")
            if cid:
                failed_ids.add(cid)
            else:
                failed_noID.append(challenge)

    # --- Phase 3: persist under the lock ---
    # Re-read the schedule from disk so that any entries added concurrently
    # (e.g. by /daily_challenge) are preserved.  We only remove entries whose
    # IDs we know were processed in Phase 1; entries not seen in Phase 1
    # (i.e. concurrently added) are kept unconditionally.
    async with _SCHEDULE_LOCK:
        fresh_schedule = await asyncio.to_thread(_load_schedule)
        merged = []
        for c in fresh_schedule:
            cid = c.get("id")
            if cid is None:
                # Legacy entry without an ID: fall back to Phase 1 classification.
                if c in remaining or c in failed_noID:
                    merged.append(c)
            elif cid not in phase1_ids:
                # Entry added concurrently after Phase 1 -- keep it.
                merged.append(c)
            elif cid in failed_ids:
                # Delivery failed -- keep for retry.
                merged.append(c)
            # else: successfully sent or expired -- drop it.
        await asyncio.to_thread(_save_schedule, merged)


# ---------------------------------------------------------------------------
# /daily_challenge
# ---------------------------------------------------------------------------

@bot.slash_command(
    name="daily_challenge",
    description="Schedule a daily gesture thread in the configured forum channel",
    guild_ids=guild_ids,
)
@discord.default_permissions(manage_guild=True)
async def daily_challenge(
    ctx: discord.ApplicationContext,
    day: discord.Option(str, description='Label used in the thread title, e.g. "Day 42" or "Saturday"'),
    release_time: discord.Option(
        str,
        description='Time to post (ET), e.g. "18:00", "6pm", "6:30pm". Defaults to 6pm.',
        required=False,
        default="18:00",
    ),
    release_date: discord.Option(
        str,
        description='Date to post (ET), e.g. "2026-06-20", "June 20", "tomorrow". Defaults to today/tomorrow.',
        required=False,
        default=None,
    ),
    reference: discord.Option(
        str,
        description="Discord image URL to use as reference. Omit to pick randomly from references.json.",
        required=False,
        default=None,
    ),
    minimum_time: discord.Option(
        str,
        description='Minimum time, e.g. "10 minutes". Omit for a random 1–15 min value.',
        required=False,
        default=None,
    ),
    extra_challenge: discord.Option(
        str,
        description="Optional extra challenge text.",
        required=False,
        default=None,
    ),
    description: discord.Option(
        str,
        description="Optional notes (stored but not shown in the post).",
        required=False,
        default=None,
    ),
    channel_id: discord.Option(
        str,
        description="Channel ID to post in. Overrides the server default (useful for test channels).",
        required=False,
        default=None,
    ),
):
    await ctx.defer(ephemeral=True)

    try:
        post_at_iso = _parse_release_datetime(release_time, release_date)
    except ValueError:
        await ctx.followup.send(
            f"Could not parse date/time `{release_date} {release_time}`. "
            "Time: `18:00`, `6pm`, `6:30pm`. Date: `2026-06-20`, `June 20`, `tomorrow`.",
            ephemeral=True,
        )
        return

    if minimum_time is None:
        minimum_time = _random_minimum_time()

    if reference is None:
        refs = await asyncio.to_thread(_load_references)
        if refs:
            reference = random.choice(refs)

    if not channel_id:
        default_channel_id = _get_guild_channel(ctx.guild_id)
        if not default_channel_id:
            await ctx.followup.send(
                "No forum channel configured for this server. "
                "An admin must run `/set_daily_channel` first, or provide a `channel_id`.",
                ephemeral=True,
            )
            return

    challenge = {
        "id":              str(uuid.uuid4()),
        "guild_id":        str(ctx.guild_id),
        "day":             day,
        "description":     description,
        "post_at":         post_at_iso,
        "reference":       reference,
        "minimum_time":    minimum_time,
        "extra_challenge": extra_challenge,
    }
    if channel_id:
        challenge["channel_id"] = channel_id

    rendered_len = len(_format_daily_post(challenge))
    if rendered_len > DISCORD_CONTENT_LIMIT:
        await ctx.followup.send(
            f"Combined post is {rendered_len} characters, exceeding Discord's "
            f"{DISCORD_CONTENT_LIMIT}-char limit. Shorten the reference, "
            f"extra_challenge, or other fields and try again.",
            ephemeral=True,
        )
        return

    async with _SCHEDULE_LOCK:
        schedule = await asyncio.to_thread(_load_schedule)
        schedule.append(challenge)
        await asyncio.to_thread(_save_schedule, schedule)

    post_at_dt = datetime.fromisoformat(post_at_iso)
    hour = post_at_dt.hour % 12 or 12
    ampm = "AM" if post_at_dt.hour < 12 else "PM"
    formatted_time = (
        f"{hour}:{post_at_dt.minute:02d} {ampm} ET, "
        f"{post_at_dt.strftime('%B')} {post_at_dt.day}, {post_at_dt.year}"
    )
    await ctx.followup.send(
        f"✓ **{day}** scheduled for **{formatted_time}**.",
        ephemeral=True,
    )


# ---------------------------------------------------------------------------
# Schedule management helpers & commands
# ---------------------------------------------------------------------------

async def _challenge_autocomplete(ctx: discord.AutocompleteContext) -> list[discord.OptionChoice]:
    guild_id = str(ctx.interaction.guild_id)
    async with _SCHEDULE_LOCK:
        schedule = await asyncio.to_thread(_load_schedule)
    guild_challenges = sorted(
        [c for c in schedule if c.get("guild_id") == guild_id],
        key=lambda c: c.get("post_at", ""),
    )
    choices = []
    for c in guild_challenges:
        try:
            t = datetime.fromisoformat(c["post_at"]).astimezone(ET)
            hour = t.hour % 12 or 12
            ampm = "AM" if t.hour < 12 else "PM"
            time_label = f"{t.month}/{t.day} {hour}:{t.minute:02d} {ampm}"
        except (KeyError, ValueError):
            time_label = "?"
        label = f"{c['day']} — {time_label}"[:100]
        value = c.get("id") or c.get("post_at", label)
        choices.append(discord.OptionChoice(name=label, value=value))
    typed = ctx.value.lower()
    return [ch for ch in choices if typed in ch.name.lower()][:25]


@bot.slash_command(
    name="list_schedule",
    description="List all pending daily art prompts scheduled for this server (admin only)",
    guild_ids=guild_ids,
)
@discord.default_permissions(manage_guild=True)
async def list_schedule(ctx: discord.ApplicationContext):
    await ctx.defer(ephemeral=True)
    guild_id = str(ctx.guild_id)
    async with _SCHEDULE_LOCK:
        schedule = await asyncio.to_thread(_load_schedule)
    guild_challenges = sorted(
        [c for c in schedule if c.get("guild_id") == guild_id],
        key=lambda c: c.get("post_at", ""),
    )
    if not guild_challenges:
        await ctx.followup.send("No challenges scheduled for this server.", ephemeral=True)
        return
    lines = []
    for i, c in enumerate(guild_challenges, 1):
        try:
            t = datetime.fromisoformat(c["post_at"]).astimezone(ET)
            hour = t.hour % 12 or 12
            ampm = "AM" if t.hour < 12 else "PM"
            time_str = f"{t.strftime('%B')} {t.day} at {hour}:{t.minute:02d} {ampm} ET"
        except (KeyError, ValueError):
            time_str = "unknown time"
        desc = c.get("description") or ""
        snippet = desc[:80] + ("…" if len(desc) > 80 else "")
        lines.append(f"**{i}.** **{c.get('day', '?')}** — posts {time_str}\n   _{snippet}_")
    body = "\n\n".join(lines)
    header = f"**Scheduled challenges ({len(guild_challenges)}):**\n\n"
    max_body = 1900 - len(header)
    if len(body) > max_body:
        body = body[:max_body].rsplit("\n", 1)[0] + "\n\n…(truncated)"
    await ctx.followup.send(header + body, ephemeral=True)


@bot.slash_command(
    name="delete_challenge",
    description="Delete a pending daily art prompt from the schedule (admin only)",
    guild_ids=guild_ids,
)
@discord.default_permissions(manage_guild=True)
async def delete_challenge(
    ctx: discord.ApplicationContext,
    challenge: discord.Option(
        str,
        description="The challenge to delete",
        autocomplete=_challenge_autocomplete,
    ),
):
    await ctx.defer(ephemeral=True)
    guild_id = str(ctx.guild_id)
    async with _SCHEDULE_LOCK:
        schedule = await asyncio.to_thread(_load_schedule)
        new_schedule = [
            c for c in schedule
            if not (
                c.get("guild_id") == guild_id
                and (c.get("id") == challenge or c.get("post_at") == challenge)
            )
        ]
        if len(new_schedule) == len(schedule):
            await ctx.followup.send(
                "Challenge not found. It may have already been posted or deleted.",
                ephemeral=True,
            )
            return
        await asyncio.to_thread(_save_schedule, new_schedule)
    await ctx.followup.send("✓ Challenge deleted from the schedule.", ephemeral=True)


@bot.slash_command(
    name="edit_challenge",
    description="Edit a pending daily art prompt in the schedule (admin only)",
    guild_ids=guild_ids,
)
@discord.default_permissions(manage_guild=True)
async def edit_challenge(
    ctx: discord.ApplicationContext,
    challenge: discord.Option(
        str,
        description="The challenge to edit",
        autocomplete=_challenge_autocomplete,
    ),
    new_day: discord.Option(
        str,
        description='New label, e.g. "Day 43"',
        required=False,
        default=None,
    ),
    description: discord.Option(
        str,
        description="New prompt description",
        required=False,
        default=None,
    ),
    release_time: discord.Option(
        str,
        description='New post time (ET), e.g. "18:00", "6pm"',
        required=False,
        default=None,
    ),
    reference: discord.Option(
        str,
        description="New reference image URL",
        required=False,
        default=None,
    ),
    minimum_time: discord.Option(
        str,
        description='New minimum time, e.g. "10 minutes"',
        required=False,
        default=None,
    ),
    extra_challenge: discord.Option(
        str,
        description="New extra challenge text",
        required=False,
        default=None,
    ),
):
    await ctx.defer(ephemeral=True)

    if all(v is None for v in (new_day, description, release_time, reference, minimum_time, extra_challenge)):
        await ctx.followup.send("No changes provided.", ephemeral=True)
        return

    if release_time is not None:
        try:
            _parse_release_datetime(release_time)  # validate only; date computed below
        except ValueError:
            await ctx.followup.send(
                f"Could not parse release time `{release_time}`. "
                "Use formats like `18:00`, `6pm`, or `6:30pm`.",
                ephemeral=True,
            )
            return

    guild_id = str(ctx.guild_id)
    async with _SCHEDULE_LOCK:
        schedule = await asyncio.to_thread(_load_schedule)
        target = next(
            (
                c for c in schedule
                if c.get("guild_id") == guild_id
                and (c.get("id") == challenge or c.get("post_at") == challenge)
            ),
            None,
        )
        if target is None:
            await ctx.followup.send(
                "Challenge not found. It may have already been posted or deleted.",
                ephemeral=True,
            )
            return
        if new_day is not None:
            target["day"] = new_day
        if description is not None:
            target["description"] = description
        if release_time is not None:
            existing_dt = datetime.fromisoformat(target["post_at"]).astimezone(ET)
            target["post_at"] = _parse_release_datetime(
                release_time, existing_dt.strftime("%Y-%m-%d")
            )
        if reference is not None:
            target["reference"] = reference
        if minimum_time is not None:
            target["minimum_time"] = minimum_time
        if extra_challenge is not None:
            target["extra_challenge"] = extra_challenge

        rendered_len = len(_format_daily_post(target))
        if rendered_len > DISCORD_CONTENT_LIMIT:
            await ctx.followup.send(
                f"Combined post is {rendered_len} characters, exceeding Discord's "
                f"{DISCORD_CONTENT_LIMIT}-char limit. Shorten the reference, "
                f"extra_challenge, or other fields and try again.",
                ephemeral=True,
            )
            return

        await asyncio.to_thread(_save_schedule, schedule)

    await ctx.followup.send("✓ Challenge updated.", ephemeral=True)


# ---------------------------------------------------------------------------
# /set_daily_channel
# ---------------------------------------------------------------------------

@bot.slash_command(
    name="set_daily_channel",
    description="Set the forum channel where daily art prompts will be posted (admin only)",
    guild_ids=guild_ids,
)
@discord.default_permissions(manage_guild=True)
async def set_daily_channel(
    ctx: discord.ApplicationContext,
    channel: discord.Option(discord.ForumChannel, description="The forum channel to post daily prompts in"),
):
    await _set_guild_channel(ctx.guild_id, str(channel.id))
    await ctx.respond(
        f"✓ Daily challenge channel set to {channel.mention}.",
        ephemeral=True,
    )


# ---------------------------------------------------------------------------
# /set_daily_role
# ---------------------------------------------------------------------------

@bot.slash_command(
    name="set_daily_role",
    description="Set the role pinged in daily art prompt posts for this server (admin only)",
    guild_ids=guild_ids,
)
@discord.default_permissions(manage_guild=True)
async def set_daily_role(
    ctx: discord.ApplicationContext,
    role: discord.Option(discord.Role, description="Role to ping when a daily prompt is posted"),
):
    if role.id == ctx.guild_id:
        await ctx.respond(
            "Can't use @everyone as the daily-ping role. Pick a non-default role.",
            ephemeral=True,
        )
        return
    await _set_guild_daily_role(ctx.guild_id, str(role.id))
    await ctx.respond(
        f"✓ Daily prompt will now ping {role.mention}.",
        ephemeral=True,
    )


# ---------------------------------------------------------------------------
# /set_required_role
# ---------------------------------------------------------------------------

@bot.slash_command(
    name="set_required_role",
    description="Set the role required to use this bot in this server (admin only)",
    guild_ids=guild_ids,
)
@discord.default_permissions(manage_guild=True)
async def set_required_role(
    ctx: discord.ApplicationContext,
    role: discord.Option(discord.Role, description="Role required to use the bot. Admins always bypass this."),
):
    await _set_guild_required_role(ctx.guild_id, str(role.id))
    await ctx.respond(
        f"✓ Bot access restricted to {role.mention} (and admins) in this server.",
        ephemeral=True,
    )


if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN not set. Copy .env.example to .env and fill it in.")
    bot.run(TOKEN)
