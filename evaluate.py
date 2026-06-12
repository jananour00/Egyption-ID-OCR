"""
Egyptian National ID OCR — Performance Metrics
================================================
Computes Character Error Rate (CER) and Word Error Rate (WER)
against a ground-truth CSV.

Usage:
    python evaluate.py --test_csv tests/ground_truth.csv --images_dir samples/

CSV format (UTF-8):
    image_filename,name,address,national_id
    id_001.jpg,أحمد محمد علي,القاهرة شارع التحرير,29801234567890
    ...
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.preprocessor import IDCardPreprocessor
from src.ocr_engine import IDCardOCR
from src.postprocessor import PostProcessor


# ──────────────────────────────────────────────────────────────
# Edit-distance helpers
# ──────────────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────

def run_evaluation(
    test_csv: str,
    images_dir: str,
    engine: str = "easyocr",
    gpu: bool = False,
) -> dict:
    rows = list(csv.DictReader(open(test_csv, encoding="utf-8")))
    if not rows:
        raise ValueError("Empty CSV")

    preprocessor = IDCardPreprocessor()
    ocr = IDCardOCR(engine=engine, gpu=gpu)
    post = PostProcessor()

    results = []
    for row in rows:
        img_path = Path(images_dir) / row["image_filename"]
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        try:
            pre = preprocessor.process(str(img_path))
            ocr_result = ocr.extract(pre["final"])
            response = post.process(ocr_result)
            data = response["data"]

            name_cer = cer(row.get("name", ""), data.get("name") or "")
            name_wer = wer(row.get("name", ""), data.get("name") or "")
            addr_cer = cer(row.get("address", ""), data.get("address") or "")
            addr_wer = wer(row.get("address", ""), data.get("address") or "")
            nid_acc = 1.0 if data.get("national_id") == row.get("national_id") else 0.0

            results.append({
                "file": row["image_filename"],
                "name_cer": name_cer,
                "name_wer": name_wer,
                "address_cer": addr_cer,
                "address_wer": addr_wer,
                "nid_exact_match": nid_acc,
                "overall_success": response["success"],
            })

        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
            results.append({"file": row["image_filename"], "error": str(e)})

    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "No valid results", "details": results}

    summary = {
        "n_images": len(valid),
        "avg_name_cer": round(sum(r["name_cer"] for r in valid) / len(valid), 4),
        "avg_name_wer": round(sum(r["name_wer"] for r in valid) / len(valid), 4),
        "avg_address_cer": round(sum(r["address_cer"] for r in valid) / len(valid), 4),
        "avg_address_wer": round(sum(r["address_wer"] for r in valid) / len(valid), 4),
        "nid_accuracy": round(sum(r["nid_exact_match"] for r in valid) / len(valid), 4),
        "overall_success_rate": round(sum(r["overall_success"] for r in valid) / len(valid), 4),
        "per_image": results,
    }
    return summary


def main():
    p = argparse.ArgumentParser(description="Evaluate OCR pipeline accuracy")
    p.add_argument("--test_csv", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--engine", default="easyocr", choices=["easyocr", "paddle"])
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--output", default="evaluation_report.json")
    args = p.parse_args()

    print(f"[INFO] Evaluating on {args.test_csv} ...")
    summary = run_evaluation(args.test_csv, args.images_dir, args.engine, args.gpu)

    print("\n── Evaluation Summary ──────────────────────")
    print(f"  Images evaluated  : {summary.get('n_images', 0)}")
    print(f"  Name  CER         : {summary.get('avg_name_cer', 'N/A'):.2%}")
    print(f"  Name  WER         : {summary.get('avg_name_wer', 'N/A'):.2%}")
    print(f"  Address CER       : {summary.get('avg_address_cer', 'N/A'):.2%}")
    print(f"  Address WER       : {summary.get('avg_address_wer', 'N/A'):.2%}")
    print(f"  NID exact match   : {summary.get('nid_accuracy', 'N/A'):.2%}")
    print(f"  Overall success   : {summary.get('overall_success_rate', 'N/A'):.2%}")
    print("────────────────────────────────────────────")

    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[INFO] Full report saved to {args.output}")


if __name__ == "__main__":
    main()
