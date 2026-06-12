"""
Egyptian National ID Card — Image Pre-processing Pipeline
==========================================================
Handles: Perspective Transformation, Denoising, Binarization,
         and Orientation Correction using OpenCV.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order corner points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left  → smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right → largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective warp using four corner points."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


# ──────────────────────────────────────────────────────────────
# Stage 1 — Perspective Correction
# ──────────────────────────────────────────────────────────────

def detect_card_and_warp(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Detect the ID card in the image and apply a perspective warp
    to produce a flat, top-down view.

    Returns:
        (warped_image, success_flag)
    """
    orig = image.copy()
    h, w = image.shape[:2]

    # ── Pre-process for edge detection ──
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # Dilate to close gaps in card outline
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=2)

    # ── Find contours & pick the largest quadrilateral ──
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    card_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (h * w * 0.1):
            card_contour = approx
            break

    if card_contour is None:
        # Fallback: use entire image as-is
        return image, False

    pts = card_contour.reshape(4, 2).astype("float32")
    warped = _four_point_transform(orig, pts)
    return warped, True


# ──────────────────────────────────────────────────────────────
# Stage 1.5 — Coarse Orientation Correction (90° / 180° / 270°)
# ──────────────────────────────────────────────────────────────

# Egyptian National ID cards are landscape with an aspect ratio ~1.585:1
# (ISO/IEC 7810 ID-1 format, like a credit card).
_ID_CARD_ASPECT_RATIO = 85.6 / 54.0  # ≈ 1.585


def correct_coarse_orientation(image: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Ensure the card is in landscape orientation.

    Phone photos are very often taken in portrait mode while the physical
    card is landscape — this results in a 90°/270° rotation that the
    fine-grained `deskew()` step (±15°) cannot fix.

    This function rotates the image by 90° if its aspect ratio indicates
    a portrait orientation, so the final image is always landscape
    (width > height), matching the real card proportions.

    Args:
        image: Image after perspective warp (or the raw image if warp failed).

    Returns:
        (corrected_image, rotation_applied_degrees)
        rotation_applied_degrees is one of {0, 90, 270} — the rotation
        that was applied to reach a landscape orientation (counter-clockwise
        positive, matching cv2.ROTATE_* conventions used internally).
    """
    h, w = image.shape[:2]

    if w >= h:
        # Already landscape (or square) — nothing to do.
        return image, 0

    # Portrait → rotate to landscape. Without running OCR we can't yet tell
    # whether the correct rotation is 90° CW or 90° CCW, so we pick CCW
    # (cv2.ROTATE_90_COUNTERCLOCKWISE) as a sensible default. The remaining
    # 180° ambiguity (upside-down landscape) is resolved later by
    # `IDCardOCR.extract_best_orientation`, which compares OCR confidence
    # across 0°/180° candidates.
    rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated, 270  # rotating CCW by 90° is equivalent to +270°


def rotate_180(image: np.ndarray) -> np.ndarray:
    """Rotate an image by 180° (used for OCR-confidence-based disambiguation)."""
    return cv2.rotate(image, cv2.ROTATE_180)




def denoise_and_binarize(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert to grayscale, denoise, and apply adaptive thresholding.

    Returns:
        (gray_image, binary_image)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # ── Non-local means denoising (excellent for scanned docs) ──
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # ── Adaptive Thresholding: handles uneven lighting ──
    binary = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    # ── Morphological cleanup: remove noise speckles ──
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    return gray, binary


# ──────────────────────────────────────────────────────────────
# Stage 3 — Orientation Correction (Deskew)
# ──────────────────────────────────────────────────────────────

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct tilt so text rows are horizontally aligned.
    Works on either a grayscale or binary image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Invert if needed (text should be white on black for moments)
    if np.mean(gray) > 127:
        gray_inv = cv2.bitwise_not(gray)
    else:
        gray_inv = gray.copy()

    coords = np.column_stack(np.where(gray_inv > 0))
    if len(coords) < 50:
        return image  # Not enough content to detect angle

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect angle quirks: values between -90 and 0
    if angle < -45:
        angle = 90 + angle

    # Small angles only (±15°); larger may indicate portrait orientation
    if abs(angle) > 15:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


# ──────────────────────────────────────────────────────────────
# Public API — Full Pipeline
# ──────────────────────────────────────────────────────────────

class IDCardPreprocessor:
    """End-to-end image pre-processing for Egyptian National ID cards."""

    def __init__(self, debug_output_dir: Optional[str] = None):
        self.debug_dir = Path(debug_output_dir) if debug_output_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _save_debug(self, name: str, img: np.ndarray):
        if self.debug_dir is not None:
            cv2.imwrite(str(self.debug_dir / name), img)

    def process(self, image_path: str) -> dict:
        """
        Run the full pre-processing pipeline.

        Args:
            image_path: Path to a JPG or PNG image.

        Returns:
            {
                "original":    np.ndarray,
                "warped":      np.ndarray,   # after perspective correction
                "oriented":    np.ndarray,   # after coarse 90/180/270 rotation fix
                "gray":        np.ndarray,   # grayscale + denoised
                "binary":      np.ndarray,   # binarized
                "final":       np.ndarray,   # deskewed binary — ready for OCR
                "warp_success": bool,
                "rotation_applied": int,     # 0, 90, or 270 — coarse rotation applied
            }
        """
        raw = cv2.imread(str(image_path))
        if raw is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Resize to manageable size while keeping aspect ratio
        h, w = raw.shape[:2]
        scale = min(1.0, 1600 / max(h, w))
        if scale < 1.0:
            raw = cv2.resize(raw, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        self._save_debug("0_original.jpg", raw)

        # Stage 1: Perspective warp
        warped, warp_ok = detect_card_and_warp(raw)
        self._save_debug("1_warped.jpg", warped)

        # Stage 1.5: Coarse orientation correction (90°/270° — handles
        # portrait-mode phone photos of a landscape card)
        oriented, rotation_applied = correct_coarse_orientation(warped)
        self._save_debug("2_oriented.jpg", oriented)

        # Stage 2: Denoise + binarize
        gray, binary = denoise_and_binarize(oriented)
        self._save_debug("3_gray.jpg", gray)
        self._save_debug("4_binary.jpg", binary)

        # Stage 3: Deskew (fine angle, ±15°)
        final = deskew(binary)
        self._save_debug("5_final_deskewed.jpg", final)

        return {
            "original":         raw,
            "warped":           warped,
            "oriented":         oriented,
            "gray":             gray,
            "binary":           binary,
            "final":            final,
            "warp_success":     warp_ok,
            "rotation_applied": rotation_applied,
        }
