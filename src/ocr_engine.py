"""
Egyptian National ID — OCR Engine
===================================
Supports EasyOCR (primary) with a PaddleOCR fallback.
Handles Arabic text (RTL) and both Western & Eastern Arabic numerals.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────
# Arabic ↔ numeral helpers
# ──────────────────────────────────────────────────────────────

# Eastern Arabic numeral mapping  ٠١٢٣٤٥٦٧٨٩
_EASTERN_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
# Persian variants
_PERSIAN_TO_WESTERN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def normalize_numerals(text: str) -> str:
    """Convert Eastern/Persian Arabic numerals to Western (ASCII) digits."""
    return text.translate(_EASTERN_TO_WESTERN).translate(_PERSIAN_TO_WESTERN)


def is_arabic_text(text: str) -> bool:
    """Return True if the majority of alphabetic chars are Arabic."""
    arabic_count = sum(
        1 for ch in text
        if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F"
    )
    alpha_count = sum(1 for ch in text if ch.isalpha())
    if alpha_count == 0:
        return False
    return arabic_count / alpha_count >= 0.6


# ──────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    raw_text: str                           # all text as one block
    lines: list[str] = field(default_factory=list)
    name: Optional[str] = None
    address: Optional[str] = None
    national_id: Optional[str] = None
    confidence: float = 0.0
    engine_used: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "address":     self.address,
            "national_id": self.national_id,
            "confidence":  round(self.confidence, 4),
            "engine":      self.engine_used,
            "raw_text":    self.raw_text,
        }


# ──────────────────────────────────────────────────────────────
# Base engine interface
# ──────────────────────────────────────────────────────────────

class BaseOCREngine:
    def read(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        """Return list of (bounding_box, text, confidence)."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# EasyOCR engine
# ──────────────────────────────────────────────────────────────

class EasyOCREngine(BaseOCREngine):
    """Wraps EasyOCR with Arabic + English language support."""

    def __init__(self, gpu: bool = False):
        import easyocr  # type: ignore
        self._reader = easyocr.Reader(
            ["ar", "en"],
            gpu=gpu,
            verbose=False,
        )

    def read(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        results = self._reader.readtext(image, detail=1, paragraph=False)
        # EasyOCR returns (bbox, text, conf)
        return [(bbox, txt, conf) for bbox, txt, conf in results]


# ──────────────────────────────────────────────────────────────
# PaddleOCR engine (fallback)
# ──────────────────────────────────────────────────────────────

class PaddleOCREngine(BaseOCREngine):
    """Wraps PaddleOCR with Arabic language support."""

    def __init__(self, use_gpu: bool = False):
        from paddleocr import PaddleOCR  # type: ignore
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ar",
            use_gpu=use_gpu,
            show_log=False,
        )

    def read(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        result = self._ocr.ocr(image, cls=True)
        output = []
        if result and result[0]:
            for line in result[0]:
                bbox, (text, conf) = line
                output.append((bbox, text, float(conf)))
        return output


# ──────────────────────────────────────────────────────────────
# OCR Orchestrator
# ──────────────────────────────────────────────────────────────

def _load_engine(engine_name: str, gpu: bool = False) -> BaseOCREngine:
    if engine_name == "easyocr":
        return EasyOCREngine(gpu=gpu)
    elif engine_name == "paddle":
        return PaddleOCREngine(use_gpu=gpu)
    else:
        raise ValueError(f"Unknown engine: {engine_name!r}. Choose 'easyocr' or 'paddle'.")


class IDCardOCR:
    """
    Runs OCR on a pre-processed ID card image and returns
    structured Name / Address / National ID fields.
    """

    def __init__(self, engine: str = "easyocr", gpu: bool = False):
        self._engine = _load_engine(engine, gpu)
        self._engine_name = engine

    # ── Main entry point ──

    def extract(self, image: np.ndarray) -> OCRResult:
        """
        Args:
            image: Pre-processed (binarized, deskewed) image as numpy array.

        Returns:
            OCRResult with parsed fields.
        """
        detections = self._engine.read(image)
        lines = [text for _, text, _ in detections]
        confidences = [conf for _, _, conf in detections]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        raw_text = "\n".join(lines)

        result = OCRResult(
            raw_text=raw_text,
            lines=lines,
            confidence=avg_conf,
            engine_used=self._engine_name,
        )

        result.national_id = self._extract_national_id(lines)
        result.name = self._extract_name(lines)
        result.address = self._extract_address(lines)

        return result

    # ── Orientation-aware entry point ──

    def extract_best_orientation(self, image: np.ndarray) -> tuple[OCRResult, int]:
        """
        Run OCR on both the given image and its 180° rotation, and return
        whichever yields the higher-confidence / more-complete result.

        This resolves the remaining 180° ambiguity left after
        `correct_coarse_orientation()` (which only guarantees a landscape
        orientation, not which edge is "up").

        Args:
            image: Pre-processed image, already corrected to landscape
                   (output of `correct_coarse_orientation`).

        Returns:
            (best_result, rotation_degrees) where rotation_degrees is
            0 if the original orientation was best, or 180 if the
            rotated candidate was best.
        """
        import cv2  # local import to keep this module's top-level light

        candidates = [
            (image, 0),
            (cv2.rotate(image, cv2.ROTATE_180), 180),
        ]

        scored: list[tuple[float, OCRResult, int]] = []
        for candidate_img, angle in candidates:
            result = self.extract(candidate_img)
            # Score: average OCR confidence, with a bonus if a valid-looking
            # 14-digit National ID was found (strong signal of correct orientation).
            score = result.confidence
            if result.national_id and len(re.sub(r"\D", "", normalize_numerals(result.national_id))) == 14:
                score += 0.5
            scored.append((score, result, angle))

        best_score, best_result, best_angle = max(scored, key=lambda t: t[0])
        return best_result, best_angle

    # ── Field extractors ──

    @staticmethod
    def _extract_national_id(lines: list[str]) -> Optional[str]:
        """
        Egyptian National ID is exactly 14 digits.
        Checks for Eastern Arabic numerals too.
        """
        for line in lines:
            normalized = normalize_numerals(line)
            # Remove spaces / dashes between digit groups
            digits_only = re.sub(r"[\s\-_]", "", normalized)
            match = re.search(r"\b(\d{14})\b", digits_only)
            if match:
                return match.group(1)
        # Fallback: any 14-digit sequence in entire block
        full = normalize_numerals(" ".join(lines))
        full_clean = re.sub(r"[^\d]", "", full)
        if len(full_clean) >= 14:
            # Try sliding window for run-together digits
            for i in range(len(full_clean) - 13):
                candidate = full_clean[i: i + 14]
                if _validate_egyptian_id(candidate):
                    return candidate
        return None

    @staticmethod
    def _extract_name(lines: list[str]) -> Optional[str]:
        """
        Looks for Arabic text lines that follow a label like الاسم / الاسم الكامل.
        Falls back to the longest Arabic line in the top half.
        """
        label_pattern = re.compile(r"(الاسم|الاسم\s*الكامل|الأسم)", re.UNICODE)
        for i, line in enumerate(lines):
            if label_pattern.search(line):
                # The name is usually the next non-empty line
                for j in range(i + 1, min(i + 3, len(lines))):
                    candidate = lines[j].strip()
                    if candidate and is_arabic_text(candidate) and len(candidate) > 3:
                        return _clean_arabic_text(candidate)
        # Heuristic: top-third lines that are Arabic
        top_lines = lines[: max(1, len(lines) // 3)]
        arabic_lines = [l for l in top_lines if is_arabic_text(l) and len(l.strip()) > 5]
        if arabic_lines:
            return _clean_arabic_text(max(arabic_lines, key=len))
        return None

    # Labels that mark the START of OTHER fields — used to stop
    # address collection before it swallows the next field's label.
    _OTHER_FIELD_LABELS = re.compile(
        r"(الرقم\s*القومي|تاريخ\s*الميلاد|النوع|الديانة|الحالة\s*الاجتماعية|"
        r"تاريخ\s*الإصدار|تاريخ\s*الانتهاء|المهنة|الاسم|الأسم)",
        re.UNICODE,
    )

    @staticmethod
    def _extract_address(lines: list[str]) -> Optional[str]:
        """
        Looks for address label العنوان / محل الإقامة then collects
        one or two following lines, stopping at the next field label.
        """
        label_pattern = re.compile(r"(العنوان|محل\s*الإقامة|العنوان\s*بالتفصيل)", re.UNICODE)
        for i, line in enumerate(lines):
            if label_pattern.search(line):
                parts = []
                for j in range(i + 1, min(i + 4, len(lines))):
                    candidate = lines[j].strip()
                    if not candidate:
                        continue
                    if label_pattern.search(candidate) or IDCardOCR._OTHER_FIELD_LABELS.search(candidate):
                        break
                    # Stop if line is purely numeric (likely the National ID)
                    if re.fullmatch(r"[\d\s\u0660-\u0669]+", normalize_numerals(candidate)):
                        break
                    parts.append(_clean_arabic_text(candidate))
                if parts:
                    return " / ".join(parts)
        # Heuristic: middle-third Arabic lines that look like addresses
        mid = lines[len(lines) // 3: 2 * len(lines) // 3]
        arabic_mid = [l for l in mid if is_arabic_text(l) and len(l.strip()) > 5]
        if len(arabic_mid) >= 2:
            return _clean_arabic_text(" ".join(arabic_mid[:2]))
        return None


# ──────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────

def _validate_egyptian_id(nid: str) -> bool:
    """
    Structural check for Egyptian National ID (14 digits):
    - Digit 1  : century (2=19xx, 3=20xx)
    - Digits 2-7: birth date YYMMDD
    - Digits 8-9: governorate code (01–35)
    - Digits 10-13: sequence
    - Digit 14 : check digit (odd=male, even=female)
    """
    if not re.fullmatch(r"\d{14}", nid):
        return False
    century = int(nid[0])
    if century not in (2, 3):
        return False
    gov_code = int(nid[7:9])
    if not (1 <= gov_code <= 35):
        return False
    return True


def _clean_arabic_text(text: str) -> str:
    """Remove non-Arabic/space characters and normalise whitespace."""
    # Keep Arabic letters, tashkeel, spaces, and basic punctuation
    cleaned = re.sub(
        r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s،,.]",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", cleaned).strip()
