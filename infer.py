#!/usr/bin/env python3
"""
Egyptian National ID OCR — Command-line Inference Script
=========================================================
Usage:
    python infer.py --image path/to/id_card.jpg
    python infer.py --image path/to/id_card.jpg --engine paddle --debug
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from src.preprocessor import IDCardPreprocessor
from src.ocr_engine import IDCardOCR
from src.postprocessor import PostProcessor


def parse_args():
    p = argparse.ArgumentParser(description="Egyptian National ID OCR Inference")
    p.add_argument("--image", required=True, help="Path to JPG/PNG image of ID card")
    p.add_argument(
        "--engine",
        choices=["easyocr", "paddle"],
        default="easyocr",
        help="OCR engine to use (default: easyocr)",
    )
    p.add_argument("--gpu", action="store_true", help="Use GPU if available")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate pre-processing images to ./debug/",
    )
    p.add_argument(
        "--output",
        help="Save JSON result to this path (optional)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Processing: {image_path}")

    # ── Pre-processing ──
    debug_dir = "debug" if args.debug else None
    preprocessor = IDCardPreprocessor(debug_output_dir=debug_dir)
    preprocess_result = preprocessor.process(str(image_path))

    warp_status = "✓ applied" if preprocess_result["warp_success"] else "✗ fallback (card edges not detected)"
    print(f"[INFO] Perspective warp: {warp_status}")

    rot = preprocess_result["rotation_applied"]
    rot_status = f"{rot}° (portrait photo corrected to landscape)" if rot else "0° (already landscape)"
    print(f"[INFO] Coarse orientation correction: {rot_status}")

    if args.debug:
        print(f"[INFO] Debug images saved to ./debug/")

    # ── OCR ──
    print(f"[INFO] Running OCR with engine: {args.engine}")
    ocr = IDCardOCR(engine=args.engine, gpu=args.gpu)
    ocr_result, flip_180 = ocr.extract_best_orientation(preprocess_result["final"])
    if flip_180:
        print(f"[INFO] OCR-confidence check: image was upside-down — applied 180° flip")

    # ── Post-processing & Validation ──
    post = PostProcessor()
    response = post.process(ocr_result)

    # ── Print results ──
    print("\n" + "=" * 55)
    print("  EXTRACTION RESULT")
    print("=" * 55)
    data = response["data"]
    val = response["validation"]

    print(f"  Name        : {data['name'] or '(not found)'}")
    print(f"  Address     : {data['address'] or '(not found)'}")
    print(f"  National ID : {data['national_id'] or '(not found)'}")
    print(f"  Confidence  : {data['confidence']:.1%}")
    print(f"  Engine      : {data['engine']}")
    print(f"  Overall OK  : {'✓ Yes' if response['success'] else '✗ No'}")

    if not response["success"]:
        print("\n  Validation Issues:")
        for field in ("national_id", "name", "address"):
            if not val[field]["valid"]:
                print(f"    • {field}: {val[field]['error']}")

    print("=" * 55)

    # ── Optional JSON output ──
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(response, ensure_ascii=False, indent=2))
        print(f"\n[INFO] JSON saved to {out_path}")

    return 0 if response["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
