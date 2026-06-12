"""
Unit tests for post-processing logic (validators, cleaners, numeral conversion).
These tests do NOT require OCR models — they test pure logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr_engine import normalize_numerals, is_arabic_text, _validate_egyptian_id
from src.postprocessor import (
    validate_national_id,
    validate_arabic_name,
    validate_address,
    clean_ocr_text,
)
from src.preprocessor import correct_coarse_orientation
import numpy as np


# ──────────────────────────────────────────────────────────────
# Coarse orientation correction (90°/270°)
# ──────────────────────────────────────────────────────────────

def test_landscape_image_unchanged():
    img = np.zeros((400, 700, 3), dtype=np.uint8)  # h < w → landscape
    corrected, rotation = correct_coarse_orientation(img)
    assert corrected.shape == img.shape
    assert rotation == 0


def test_portrait_image_rotated_to_landscape():
    img = np.zeros((700, 400, 3), dtype=np.uint8)  # h > w → portrait
    corrected, rotation = correct_coarse_orientation(img)
    h, w = corrected.shape[:2]
    assert w > h, "Portrait image should be rotated to landscape"
    assert rotation == 270


def test_square_image_unchanged():
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    corrected, rotation = correct_coarse_orientation(img)
    assert rotation == 0
    assert corrected.shape == img.shape


# ──────────────────────────────────────────────────────────────
# Numeral normalization
# ──────────────────────────────────────────────────────────────

def test_eastern_arabic_numerals_converted():
    assert normalize_numerals("٢٩٨٠١٢٣٤٥٦٧٨٩٠") == "29801234567890"


def test_mixed_numerals_converted():
    assert normalize_numerals("12٣45") == "12345"


def test_western_numerals_unchanged():
    assert normalize_numerals("29801234567890") == "29801234567890"


# ──────────────────────────────────────────────────────────────
# Arabic text detection
# ──────────────────────────────────────────────────────────────

def test_arabic_text_detected():
    assert is_arabic_text("أحمد محمد علي") is True


def test_english_text_not_arabic():
    assert is_arabic_text("Ahmed Mohamed") is False


def test_mixed_mostly_arabic():
    assert is_arabic_text("أحمد Ahmed") is True


# ──────────────────────────────────────────────────────────────
# National ID validation
# ──────────────────────────────────────────────────────────────

def test_valid_national_id():
    # Century=2 (1900s), governorate=01 (Cairo)
    valid, err = validate_national_id("29005010123456")
    assert valid is True
    assert err is None


def test_national_id_wrong_length():
    valid, err = validate_national_id("1234567890")
    assert valid is False
    assert "14 digits" in err


def test_national_id_with_letters():
    valid, err = validate_national_id("2900501A123456")
    assert valid is False
    assert "non-digit" in err


def test_national_id_eastern_numerals():
    # Same as valid example but in Eastern Arabic numerals
    valid, err = validate_national_id("٢٩٠٠٥٠١٠١٢٣٤٥٦")
    assert valid is True


def test_national_id_invalid_century():
    # Century digit '5' is invalid (must be 2 or 3)
    valid, err = validate_national_id("59005010123456")
    assert valid is False
    assert "structural" in err.lower()


def test_national_id_missing():
    valid, err = validate_national_id(None)
    assert valid is False
    assert "missing" in err.lower()


# ──────────────────────────────────────────────────────────────
# Name validation
# ──────────────────────────────────────────────────────────────

def test_valid_arabic_name():
    valid, err = validate_arabic_name("أحمد محمد علي حسن")
    assert valid is True
    assert err is None


def test_name_with_digits_invalid():
    valid, err = validate_arabic_name("أحمد 123")
    assert valid is False
    assert "digits" in err


def test_name_single_word_invalid():
    valid, err = validate_arabic_name("أحمد")
    assert valid is False
    assert "two words" in err


def test_name_english_invalid():
    valid, err = validate_arabic_name("Ahmed Mohamed Ali")
    assert valid is False
    assert "non-Arabic" in err


def test_name_missing():
    valid, err = validate_arabic_name("")
    assert valid is False


# ──────────────────────────────────────────────────────────────
# Address validation
# ──────────────────────────────────────────────────────────────

def test_valid_address():
    valid, err = validate_address("القاهرة، شارع التحرير، مدينة نصر")
    assert valid is True


def test_address_too_short():
    valid, err = validate_address("ABC")
    assert valid is False
    assert "short" in err


def test_address_missing():
    valid, err = validate_address(None)
    assert valid is False


# ──────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────

def test_clean_removes_security_symbols():
    dirty = "أحمد#@! محمد<>|علي"
    cleaned = clean_ocr_text(dirty)
    assert "#" not in cleaned
    assert "@" not in cleaned
    assert "<" not in cleaned


def test_clean_collapses_whitespace():
    dirty = "أحمد    محمد   علي"
    cleaned = clean_ocr_text(dirty)
    assert cleaned == "أحمد محمد علي"
