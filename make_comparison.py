#!/usr/bin/env python3
"""Tile the original input image with the multiview renders for comparison."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"D:\PythonFiles\flash3d-main")
SRC = ROOT / "data" / "DS0304.jpg"
MV_RGB = ROOT / "outputs" / "my_scene_multiview" / "rgb"
OUT = ROOT / "outputs" / "wide_full_gaussians" / "comparison.png"

W, H, LABEL_H = 384, 256, 24
COLS, ROWS = 5, 2
PAD, BG = 6, (255, 255, 255)

MV_WIDE = ROOT / "outputs" / "wide_full_gaussians" / "rgb"

ITEMS = [
    ("original", SRC),
    ("00_far_left_up", MV_WIDE / "00_far_left_up.png"),
    ("01_center_up", MV_WIDE / "01_center_up.png"),
    ("02_far_right_up", MV_WIDE / "02_far_right_up.png"),
    ("03_far_left", MV_WIDE / "03_far_left.png"),
    ("04_center", MV_WIDE / "04_center.png"),
    ("05_far_right", MV_WIDE / "05_far_right.png"),
    ("06_far_left_down", MV_WIDE / "06_far_left_down.png"),
    ("07_center_down", MV_WIDE / "07_center_down.png"),
    ("08_far_right_down", MV_WIDE / "08_far_right_down.png"),
]

canvas_w = COLS * W + (COLS + 1) * PAD
canvas_h = ROWS * (H + LABEL_H) + (ROWS + 1) * PAD
canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()

for i, (label, path) in enumerate(ITEMS):
    if not path.is_file():
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    r, c = divmod(i, COLS)
    x = PAD + c * (W + PAD)
    y = PAD + r * (H + LABEL_H + PAD)
    draw.rectangle([x, y, x + W, y + LABEL_H], fill=(235, 235, 235))
    draw.text((x + 8, y + 4), label, fill=(0, 0, 0), font=font)
    canvas.paste(img, (x, y + LABEL_H))

canvas.save(OUT)
print(f"saved {OUT} ({canvas_w}x{canvas_h})")
