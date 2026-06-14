# Painting Utilities

A Discord bot for analyzing painting images, extracting color palettes, and running daily art challenges. Useful for digital artists, designers, and painters.

## Setup

**Requirements:** Python 3.10+, dependencies in `requirements.txt`

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
DISCORD_TOKEN=your_bot_token_here

# Command registration scope — leave blank for global (all servers, up to 1 hour to propagate)
# or set to one guild ID for instant registration in that server only (useful for testing)
DISCORD_GUILD_ID=your_guild_id_here

# Optional: role to ping in daily art prompt posts
DAILY_ROLE_ID=role_id_here
```

Run the bot:

```bash
python bot.py
```

---

## Multi-server usage

The bot works in any number of servers without changes to `.env`. Remove or leave `DISCORD_GUILD_ID` blank so slash commands register globally. Then in each server:

1. Invite the bot with the appropriate permissions (slash commands, forum post creation)
2. Run `/set_daily_channel` to point the bot at that server's forum channel

Each server's channel configuration is stored independently in `config.json`. Daily challenge schedules are also per-server — running `/daily_challenge` in one server only affects that server's queue.

---

## Access control

Run `/set_required_role` in a server to restrict all bot commands to members with a specific role. This is stored per-server in `config.json`, so each server can have a different required role (or none at all). Administrators always bypass the check. Users without the required role receive an ephemeral error message.

---

## Commands

All commands are Discord slash commands. Images must be PNG or JPEG, max 15 MB.

---

### `/analyze`
Full image analysis: dominant colors, image statistics, and two chart images.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | The painting to analyze |
| `num_colors` | 10 (3–16) | Number of dominant colors to extract |
| `show_rgb` | false | Include RGB values for each color |
| `show_cmyk` | false | Include CMYK values for each color |

**Output:** Embed with dimensions, brightness, contrast, saturation, dominant hue, and palette type. Attachments: `palette.png` swatch chart and `hue_sat.png` hue/saturation distribution chart.

---

### `/palette`
Quick color palette extraction — colors only, no stats.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | The painting to analyze |
| `num_colors` | 10 (3–16) | Number of dominant colors to extract |
| `show_rgb` | false | Include RGB values for each color |
| `show_cmyk` | false | Include CMYK values for each color |

**Output:** Embed listing all colors with hex codes, names, and coverage percentages. Attachment: `palette.png` swatch chart.

---

### `/gradient_map`
Remap image tones through a color gradient. Each pixel's luminance maps to a gradient color, preserving light/dark structure while replacing original hues.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Image to process |
| `preset` | `fire` | Built-in gradient: `fire`, `ocean`, `forest`, `amethyst`, `grayscale`, `sunset`, `ice` |
| `start_color` | — | Custom shadow color as hex (e.g. `#1a0030`) — must be paired with `end_color` |
| `end_color` | — | Custom highlight color as hex (e.g. `#ffe080`) — must be paired with `start_color` |

**Output:** Attachments: `gradient_map.png` result and `gradient_swatch.png` preview strip.

---

### `/palette_gradient`
Extract the image's dominant colors, arrange them by luminance into gradient stops, then apply that gradient as a tone map back onto the image.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Image to process |
| `num_colors` | 5 (3–10) | Colors to extract for the gradient |

**Output:** Embed with hex stops. Attachments: `gradient_map.png` result and `gradient_swatch.png` preview strip.

---

### `/export_palette`
Extract dominant colors and export them as a palette file for design software.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Image to extract colors from |
| `format` | `ase` | `ase` (Adobe), `swatches` (Procreate), `gpl` (GIMP/Inkscape), `aco` (Photoshop legacy), `css` (CSS variables), `tailwind` (Tailwind config) |
| `num_colors` | 10 (3–16) | Number of colors to extract |
| `palette_name` | `Palette` | Name embedded in the palette file |

**Output:** Embed listing colors with hex codes and coverage. Attachment: palette file ready to import.

---

### `/export_gradient`
Export an image-derived gradient as a file for GIMP/Krita or JSON.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Image to extract colors from |
| `format` | `ggr` | `ggr` (GIMP/Krita) or `json` |
| `num_colors` | 5 (3–100) | Colors to extract |
| `sort_by` | `value` | How to order colors in the gradient: `value`, `luminance`, `hue`, `saturation` |
| `gradient_name` | `palette_gradient` | Name embedded in the file |
| `reverse` | false | Flip the gradient direction |

**Output:** Embed with gradient hex stops and preview strip. Attachment: `.ggr` or `.json` gradient file.

---

### `/color_info`
Look up a hex color: name, RGB, CMYK, HSV, brightness, temperature, and harmony suggestions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hex_color` | required | Hex code, e.g. `#3a7bd5` or `3a7bd5` |

**Output:** Embed with all color properties and a swatch image showing the color alongside complement, triadic, and analogous harmonies.

---

### `/compare`
Compare the dominant palettes of two images side by side.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_a` | required | First image |
| `image_b` | required | Second image |
| `num_colors` | 8 (3–16) | Colors to extract per image |

**Output:** Embed with top colors from each image and their palette types. Attachment: `compare.png` side-by-side chart.

---

### `/colorblind`
Simulate how your image looks to people with color blindness.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Image to simulate |
| `type` | `all` | `all` (4-panel comparison), `deuteranopia`, `protanopia`, or `tritanopia` |

**Output:** Attachment: simulated image or 4-panel comparison chart.

---

### `/recolor`
Apply the color palette from one image onto another.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source` | required | Image whose palette is extracted |
| `target` | required | Image to recolor |
| `num_colors` | 8 (3–16) | Colors to extract from source |

**Output:** Attachment: `recolor.png` — the target image recolored with the source palette.

---

### `/suggest_harmony`
Suggest colors that would harmonize with the existing palette in an image.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Your painting |
| `num_colors` | 8 (3–16) | Colors to extract from the image |

**Output:** Embed listing suggested harmony colors (complement, triadic, analogous). Attachment: harmony chart.

---

### `/help`
Show all available commands, or get detailed parameter info for a specific command.

```
/help                        — overview of all commands
/help command:analyze        — detailed help for /analyze
```

---

## Daily art challenge (admin only)

These commands require the Administrator permission.

### `/daily_challenge`
Schedule a formatted art prompt thread in the configured forum channel.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `day` | required | Label shown at the top of the post, e.g. `Day 42` or `Saturday` |
| `description` | required | The art prompt text |
| `release_time` | `18:00` (6 PM ET) | When to post — e.g. `18:00`, `6pm`, `6:30pm` |
| `reference` | random from `references.json` | Discord image URL to show as a reference |
| `minimum_time` | random 1–15 min | Suggested minimum time, e.g. `10 minutes` |
| `extra_challenge` | — | Optional additional challenge text |

The post is saved to `daily_schedule.json` and fired by a background task that checks every minute. It survives bot restarts. The resulting forum thread looks like:

```
DAILY ART PROMPT

Day 42
@dailyprompt
Paint a stormy coastline

□  □  □

REFERENCE
> https://...

MINIMUM TIME
> 7 minutes

EXTRA CHALLENGE
> Use only cool tones

□  □  □
```

**Random references:** add Discord CDN image URLs (one per entry) to `references.json` and they will be picked at random when no `reference` is provided.

---

### `/set_daily_channel`
Set the forum channel where daily prompts will be posted for this server. Stored in `config.json` and persists across restarts. Run this once per server after inviting the bot.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `channel` | required | The forum channel to post into |

---

### `/set_required_role`
Restrict all bot commands in this server to members with a specific role. Stored per-server in `config.json`. Administrators always bypass this check. Run without a role restriction configured and anyone can use the bot.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `role` | required | The role allowed to use the bot |
