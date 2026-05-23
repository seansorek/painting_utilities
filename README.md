# Painting Utilities

A Discord bot for analyzing painting images and extracting color palettes. Useful for digital artists, designers, and painters who want to understand the colors in their work or export palettes to professional tools.

## Setup

**Requirements:** Python 3.10+, dependencies in `requirements.txt`

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
DISCORD_TOKEN=your_bot_token_here
DISCORD_GUILD_ID=your_guild_id_here   # optional; omit for global commands
```

Run the bot:

```bash
python bot.py
```

---

## Commands

All commands are Discord slash commands. Images must be PNG or JPEG, max 15 MB.

### `/analyze`
Full image analysis: dominant colors, image statistics, and two chart images.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | The painting to analyze |
| `num_colors` | 10 (3–16) | Number of dominant colors to extract |
| `show_rgb` | false | Include RGB values for each color |
| `show_cmyk` | false | Include CMYK values for each color |

**Output:** Embed with dimensions, brightness, contrast, saturation, dominant hue range, palette type, and top 5 colors. Attachments: `palette.png` swatch chart and `hue_sat.png` hue/saturation distribution chart.

---

### `/palette`
Quick color palette extraction — no stats, just colors.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | The painting to analyze |
| `num_colors` | 10 (3–16) | Number of dominant colors to extract |
| `show_rgb` | false | Include RGB values for each color |
| `show_cmyk` | false | Include CMYK values for each color |

**Output:** Embed listing all colors with hex codes, color names, and coverage percentages. Attachment: `palette.png` swatch chart.

---

### `/gradient_map`
Remap image tones through a color gradient. Each pixel's luminance is mapped to a color from the gradient, preserving light/dark structure while replacing original colors.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Image to process |
| `preset` | `fire` | Built-in gradient: `fire`, `ocean`, `forest`, `amethyst`, `grayscale`, `sunset`, `ice` |
| `start_color` | — | Custom shadow color as hex (e.g. `#1a0030`) — must be paired with `end_color` |
| `end_color` | — | Custom highlight color as hex (e.g. `#ffe080`) — must be paired with `start_color` |

**Output:** Embed with gradient label and image dimensions. Attachments: `gradient_map.png` result and `gradient_swatch.png` preview strip.

---

### `/palette_gradient`
Extract the image's dominant colors, arrange them by luminance into gradient stops, then apply that gradient as a tone map back onto the image.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Image to process |
| `num_colors` | 5 (3–10) | Colors to extract for the gradient |

**Output:** Embed with gradient hex stops and image dimensions. Attachments: `gradient_map.png` result and `gradient_swatch.png` preview strip.

---

### `/export_palette`
Extract dominant colors and export them as a palette file for design software.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Image to extract colors from |
| `format` | `ase` | `ase` for Adobe apps (Photoshop, Illustrator, InDesign) or `swatches` for Procreate |
| `num_colors` | 10 (3–16) | Number of colors to extract |
| `palette_name` | `Palette` | Name embedded in the palette file |

**Output:** Embed listing all colors with hex codes, names, and coverage. Attachment: `palette.ase` or `palette.swatches` file ready to import.

---

### `/help`
Show all available commands, or get detailed parameter info for a specific command.

```
/help                        — overview of all commands
/help command:analyze        — detailed help for /analyze
```
