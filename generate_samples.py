"""
Synthetic Egyptian National ID Card Generator (v2)
=====================================================
Generates realistic fake (non-real) ID card images with PROPERLY RENDERED
Arabic text — using Pillow's text engine, which (on systems with libraqm,
the default in modern Pillow) handles Arabic shaping and bidi ordering
automatically, no `arabic_reshaper` / `python-bidi` dependency required.

Produces three kinds of test images:
  1. A clean, mostly-flat landscape card (baseline).
  2. A tilted landscape card on a textured background (perspective + deskew test).
  3. A PORTRAIT-photographed card, like a phone photo taken in portrait
     mode of a landscape card — tests the 90°/270° orientation correction.

Usage:
    python generate_samples.py --out samples/ --count 3
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────
# Fictional sample data (no real persons / IDs)
# ──────────────────────────────────────────────────────────────

NAMES = [
    "أحمد محمد علي حسن",
    "فاطمة محمود إبراهيم سالم",
    "محمد عبد الرحمن يوسف",
    "مريم أحمد السيد عبده",
    "خالد حسن إبراهيم محمد",
]

ADDRESSES = [
    "القاهرة شارع التحرير مدينة نصر",
    "الجيزة شارع الهرم حي الدقي",
    "الإسكندرية شارع فؤاد محطة الرمل",
    "المنصورة شارع الجمهورية وسط البلد",
    "أسيوط شارع الثورة حي الوالدية",
]

_EASTERN_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def _to_eastern(digits: str) -> str:
    return "".join(_EASTERN_DIGITS[int(d)] for d in digits)


def random_national_id() -> str:
    """Generate a structurally-valid (fictional) 14-digit Egyptian National ID."""
    century = random.choice(["2", "3"])  # 2 = 1900s, 3 = 2000s
    yy = f"{random.randint(0, 99):02d}"
    mm = f"{random.randint(1, 12):02d}"
    dd = f"{random.randint(1, 28):02d}"
    governorate = f"{random.randint(1, 35):02d}"
    sequence = f"{random.randint(0, 9999):04d}"
    check = f"{random.randint(0, 9)}"
    return century + yy + mm + dd + governorate + sequence + check


# ──────────────────────────────────────────────────────────────
# Font loading
# ──────────────────────────────────────────────────────────────

# FreeSerif ships with most Linux distros and has good Arabic coverage.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    # Last resort — bitmap default (won't render Arabic well)
    return ImageFont.load_default()


# ──────────────────────────────────────────────────────────────
# Card rendering (PIL → numpy/BGR for OpenCV)
# ──────────────────────────────────────────────────────────────

def make_card(name: str, address: str, national_id: str,
               use_eastern_numerals: bool = False) -> np.ndarray:
    """
    Render a flat, front-facing synthetic Egyptian ID card (ISO ID-1
    aspect ratio ~1.585:1) with real Arabic text.

    Returns a BGR numpy array (OpenCV format).
    """
    W, H = 1012, 638  # ~85.6mm x 54mm at ~12px/mm

    card = Image.new("RGB", (W, H), (245, 245, 248))
    draw = ImageDraw.Draw(card)

    # ── Header band ──
    draw.rectangle([0, 0, W, 95], fill=(20, 70, 65))
    title_font = _load_font(34)
    draw.text((W - 30, 25), "جمهورية مصر العربية", font=title_font, fill=(245, 245, 248), anchor="ra")
    sub_font = _load_font(22)
    draw.text((W - 30, 65), "بطاقة تحقيق شخصية", font=sub_font, fill=(220, 200, 140), anchor="ra")

    # ── Photo placeholder (left side) ──
    draw.rectangle([40, 130, 260, 400], fill=(205, 205, 205), outline=(120, 120, 120), width=2)
    photo_font = _load_font(22)
    draw.text((150, 265), "PHOTO", font=photo_font, fill=(140, 140, 140), anchor="mm")

    # ── Field labels + values (right side, RTL) ──
    label_font = _load_font(22)
    value_font = _load_font(30)
    right_edge = W - 30

    draw.text((right_edge, 130), "الاسم", font=label_font, fill=(90, 90, 90), anchor="ra")
    draw.text((right_edge, 162), name, font=value_font, fill=(20, 20, 20), anchor="ra")

    draw.text((right_edge, 230), "العنوان", font=label_font, fill=(90, 90, 90), anchor="ra")
    draw.text((right_edge, 262), address, font=value_font, fill=(20, 20, 20), anchor="ra")

    draw.text((right_edge, 330), "الرقم القومي", font=label_font, fill=(90, 90, 90), anchor="ra")
    id_display = _to_eastern(national_id) if use_eastern_numerals else national_id
    id_font = _load_font(34)
    draw.text((right_edge, 362), id_display, font=id_font, fill=(20, 20, 20), anchor="ra")

    # ── Security guilloche pattern (diagonal lines, lower area) ──
    for x in range(-H, W, 14):
        draw.line([(x, H), (x + H, 0)], fill=(225, 228, 230), width=1)

    # ── Border ──
    draw.rectangle([0, 0, W - 1, H - 1], outline=(60, 60, 60), width=3)

    return cv2.cvtColor(np.array(card), cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────────
# Scene composition: place card on a background, with tilt/rotation
# ──────────────────────────────────────────────────────────────

def place_on_background(card: np.ndarray, tilt_deg: float = 0.0,
                         portrait_photo: bool = False,
                         margin: int = 120) -> np.ndarray:
    """
    Compose the card onto a larger textured background, optionally
    tilted (simulates a slightly-rotated photo) or rendered as a
    portrait-orientation photo (simulates a phone held vertically
    while photographing a landscape card).
    """
    ch, cw = card.shape[:2]
    canvas_w, canvas_h = cw + 2 * margin, ch + 2 * margin
    canvas = np.full((canvas_h, canvas_w, 3), 38, dtype=np.uint8)

    # Simple checkered "fabric" background texture
    for gy in range(0, canvas_h, 60):
        for gx in range(0, canvas_w, 60):
            if (gx // 60 + gy // 60) % 3 == 0:
                cv2.rectangle(canvas, (gx, gy), (gx + 40, gy + 40), (55, 55, 55), -1)

    canvas[margin:margin + ch, margin:margin + cw] = card

    if tilt_deg != 0:
        M = cv2.getRotationMatrix2D((canvas_w // 2, canvas_h // 2), tilt_deg, 1.0)
        canvas = cv2.warpAffine(canvas, M, (canvas_w, canvas_h), borderValue=(38, 38, 38))

    if portrait_photo:
        # Simulate holding the phone vertically: rotate the whole scene 90°
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)

    return canvas


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate synthetic Egyptian ID card images")
    p.add_argument("--out", default="samples", help="Output directory")
    p.add_argument("--count", type=int, default=3, help="Number of cards to generate")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = ["image_filename,name,address,national_id"]

    for i in range(args.count):
        name = NAMES[i % len(NAMES)]
        address = ADDRESSES[i % len(ADDRESSES)]
        nid = random_national_id()
        use_eastern = (i % 2 == 1)  # alternate Eastern/Western numerals

        card = make_card(name, address, nid, use_eastern_numerals=use_eastern)

        # Vary the scene: tilt, or full portrait-photo rotation
        if i == 0:
            scene = place_on_background(card, tilt_deg=0)          # flat baseline
        elif i % 3 == 1:
            scene = place_on_background(card, tilt_deg=random.uniform(4, 9))  # tilted
        else:
            scene = place_on_background(card, tilt_deg=random.uniform(-3, 3),
                                          portrait_photo=True)       # portrait phone photo

        filename = f"id_{i+1:03d}.jpg"
        cv2.imwrite(str(out_dir / filename), scene)

        rows.append(f"{filename},{name},{address},{nid}")
        numeral_note = "Eastern (٠-٩)" if use_eastern else "Western (0-9)"
        print(f"[OK] {filename}  nid={nid}  numerals={numeral_note}  shape={scene.shape[:2]}")

    gt_path = out_dir.parent / "tests" / "ground_truth.csv"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"\n[INFO] Ground truth written to {gt_path}")
    print("[INFO] Cards rendered with real Arabic text via Pillow (libraqm shaping).")


if __name__ == "__main__":
    main()
