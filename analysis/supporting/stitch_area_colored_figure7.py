"""Stitch the four area-colored panels into one Figure-7-style composite:
A = descriptor dendrogram, B = descriptor PCoA, C = morphometric dendrogram,
D = morphometric PCA, matching the original Figure 7 panel layout/lettering.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SRC = Path(__file__).resolve().parents[2] / "figures" / "results" / "plots" / "area_colored_review"
OUT = SRC / "Figure7_by_area_stitched.png"

PANEL_W = 2200  # target width per column, px
LABEL_FONT_SIZE = 90


def load_scaled(name, width):
    im = Image.open(SRC / name).convert("RGB")
    scale = width / im.width
    return im.resize((width, int(im.height * scale)), Image.LANCZOS)


def add_label(im, letter):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arialbd.ttf", LABEL_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 10), letter, fill="black", font=font)
    return im


def main():
    a = add_label(load_scaled("descriptor_dendrogram_by_area.png", PANEL_W), "A")
    b = add_label(load_scaled("descriptor_pca_by_area.png", PANEL_W), "B")
    c = add_label(load_scaled("morphometric_dendrogram_by_area.png", PANEL_W), "C")
    d = add_label(load_scaled("morphometric_pca_by_area.png", PANEL_W), "D")

    row1_h = max(a.height, b.height)
    row2_h = max(c.height, d.height)
    total_w = PANEL_W * 2
    total_h = row1_h + row2_h + 40

    canvas = Image.new("RGB", (total_w, total_h), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (PANEL_W, 0))
    canvas.paste(c, (0, row1_h + 40))
    canvas.paste(d, (PANEL_W, row1_h + 40))

    canvas.save(OUT, dpi=(300, 300))
    print("written", OUT, canvas.size)


if __name__ == "__main__":
    main()
