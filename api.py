"""
Egyptian National ID OCR — FastAPI Service
===========================================
Run:  uvicorn api:app --reload --port 8000

POST /extract  → upload an image, get JSON with name/address/ID
GET  /health   → liveness check
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.preprocessor import IDCardPreprocessor
from src.ocr_engine import IDCardOCR
from src.postprocessor import PostProcessor

# ──────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Egyptian National ID OCR API",
    description=(
        "Extracts Name, Address, and 14-digit National ID from "
        "images of Egyptian National ID cards."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Lazy-load heavy models once
# ──────────────────────────────────────────────────────────────

_preprocessor: IDCardPreprocessor | None = None
_ocr: IDCardOCR | None = None
_postprocessor: PostProcessor | None = None


def get_pipeline():
    global _preprocessor, _ocr, _postprocessor
    if _preprocessor is None:
        debug_dir = os.getenv("DEBUG_OUTPUT_DIR", None)
        _preprocessor = IDCardPreprocessor(debug_output_dir=debug_dir)
        _ocr = IDCardOCR(
            engine=os.getenv("OCR_ENGINE", "easyocr"),
            gpu=os.getenv("USE_GPU", "0") == "1",
        )
        _postprocessor = PostProcessor()
    return _preprocessor, _ocr, _postprocessor


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/extract", summary="Extract ID fields from an uploaded image")
async def extract(file: UploadFile = File(..., description="JPG or PNG image of the ID card")):
    """
    Upload a JPG / PNG image of an Egyptian National ID card.

    Returns JSON:
    ```json
    {
      "data": {
        "name": "...",
        "address": "...",
        "national_id": "...",
        "confidence": 0.91,
        "engine": "easyocr"
      },
      "validation": { ... },
      "success": true,
      "processing_time_ms": 1234
    }
    ```
    """
    # ── Validate upload ──
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPG or PNG.",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum is {MAX_FILE_SIZE_MB} MB.",
        )

    t0 = time.perf_counter()

    # ── Decode image ──
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Is it a valid JPG/PNG?")

    # ── Run pipeline ──
    try:
        pre, ocr, post = get_pipeline()

        # Write to a temp file because IDCardPreprocessor expects a path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        preprocess_result = pre.process(tmp_path)
        os.unlink(tmp_path)

        ocr_result, flip_180 = ocr.extract_best_orientation(preprocess_result["final"])
        response = post.process(ocr_result)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    response["processing_time_ms"] = elapsed_ms
    response["warp_applied"] = preprocess_result.get("warp_success", False)
    response["coarse_rotation_applied"] = preprocess_result.get("rotation_applied", 0)
    response["upside_down_corrected"] = bool(flip_180)

    return JSONResponse(content=response)
