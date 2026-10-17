"""Draw the app icon: three chained video frames with a play symbol (writes tdplay/assets/AppIcon.png/.icns).

Run:  python scripts/make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "tdplay" / "assets"
S = 1024
SS = 4  # draw at 4x and downsample for smooth edges


def lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def gradient(size, top_left, bottom_right):
    w, h = size
    img = Image.new("RGBA", size)
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = lerp(top_left, bottom_right, (x / w + y / h) / 2) + (255,)
    return img


def rounded_mask(size, box, radius):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    return mask


def main() -> None:
    big = S * SS
    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))

    # macOS icon grid: an 824 px rounded square centred on the 1024 canvas, with a soft shadow.
    inset, radius = 100 * SS, 185 * SS
    box = (inset, inset, big - inset, big - inset)
    shadow = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((box[0], box[1] + 14 * SS, box[2], box[3] + 14 * SS),
                                             radius=radius, fill=(0, 0, 0, 110))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18 * SS)))
    bg = gradient((big // 8, big // 8), (24, 18, 54), (64, 22, 92)).resize((big, big), Image.BICUBIC)
    canvas.paste(bg, (0, 0), rounded_mask((big, big), box, radius))

    # Three stacked frames, back to front: the segments of one long take.
    draw = ImageDraw.Draw(canvas)
    frame_w, frame_h = 470 * SS, 300 * SS
    for i, alpha in enumerate((70, 130, 255)):
        x0 = (205 + i * 70) * SS
        y0 = (300 + i * 70) * SS
        rect = (x0, y0, x0 + frame_w, y0 + frame_h)
        if i < 2:
            draw.rounded_rectangle(rect, radius=40 * SS, outline=(255, 170, 90, alpha), width=14 * SS)
            continue
        fill = gradient((frame_w // 8, frame_h // 8), (255, 138, 40), (236, 64, 122)).resize((frame_w, frame_h))
        canvas.paste(fill, (x0, y0), rounded_mask((frame_w, frame_h), (0, 0, frame_w, frame_h), 40 * SS))
        cx, cy, r = x0 + frame_w // 2, y0 + frame_h // 2, 92 * SS
        draw.polygon([(cx - r * 0.55, cy - r), (cx - r * 0.55, cy + r), (cx + r, cy)], fill=(255, 255, 255, 245))
        # film-strip perforations along the top and bottom edges
        for k in range(7):
            hx = x0 + 38 * SS + k * 66 * SS
            for hy in (y0 + 16 * SS, y0 + frame_h - 40 * SS):
                draw.rounded_rectangle((hx, hy, hx + 30 * SS, hy + 24 * SS), radius=6 * SS, fill=(255, 255, 255, 120))

    icon = canvas.resize((S, S), Image.LANCZOS)
    OUT.mkdir(parents=True, exist_ok=True)
    icon.save(OUT / "AppIcon.png")
    icon.save(OUT / "AppIcon.icns")
    print(f"wrote {OUT / 'AppIcon.png'} and {OUT / 'AppIcon.icns'}")


if __name__ == "__main__":
    main()
