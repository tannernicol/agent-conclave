#!/usr/bin/env python3
"""Generate a deterministic demo GIF from text slides (synthetic content only)."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_frame(title: str, subtitle: str, step: str, idx: int, total: int, width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(18, 24, 34))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        shade = 18 + int(20 * y / max(height - 1, 1))
        draw.line((0, y, width, y), fill=(shade, shade + 6, shade + 10))

    title_font = load_font(48)
    subtitle_font = load_font(26)
    step_font = load_font(34)
    meta_font = load_font(22)

    draw.text((60, 48), title, fill=(240, 246, 252), font=title_font)
    draw.text((60, 120), subtitle, fill=(175, 188, 204), font=subtitle_font)

    draw.rounded_rectangle((60, 200, width - 60, height - 120), radius=18, fill=(28, 36, 48), outline=(70, 95, 125), width=2)
    draw.text((90, 250), f"Step {idx + 1}: {step}", fill=(233, 242, 251), font=step_font)

    progress_w = width - 120
    fill_w = int(progress_w * ((idx + 1) / max(total, 1)))
    draw.rounded_rectangle((60, height - 78, 60 + progress_w, height - 48), radius=8, fill=(42, 56, 75))
    draw.rounded_rectangle((60, height - 78, 60 + fill_w, height - 48), radius=8, fill=(67, 150, 255))

    draw.text((width - 220, height - 112), f"{idx + 1}/{total}", fill=(188, 203, 220), font=meta_font)
    draw.text((60, height - 112), "Synthetic demo content only", fill=(188, 203, 220), font=meta_font)
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", required=True)
    parser.add_argument("--step", action="append", default=[])
    parser.add_argument("--output", default="docs/demo.gif")
    parser.add_argument("--duration-ms", type=int, default=5000)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    steps = args.step or ["Initialize", "Run workflow", "Review results", "Export artifacts"]
    frames = [
        render_frame(args.title, args.subtitle, step, idx, len(steps), args.width, args.height)
        for idx, step in enumerate(steps)
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
