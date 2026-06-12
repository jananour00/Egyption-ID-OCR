"""
Egyptian National ID — Post-processing & Logic Validation
==========================================================
Cleans noisy OCR output and validates extracted fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.ocr_engine import OCRResult, normalize_numerals, _validate_egyptian_id, _clean_arabic_text


# ──────────────────────────────────────────────────────────────
# Validation result
# ──────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    national_id_valid: bool = False
    national_id_error: Optional[str] = None
    name_valid: bool = False
    name_error: Optional[str] = None
    address_valid: bool = False
    address_error: Optional[str] = None

    @property
    def all_valid(self) -> bool:
        return self.national_id_valid and self.name_valid and self.address_valid

    def to_dict(self) -> dict:
        return {
            "all_valid": self.all_valid,
            "national_id": {
                "valid": self.national_id_valid,
                "error": self.national_id_error,
            },
            "name": {
                "valid": self.name_valid,
                "error": self.name_error,
            },
            "address": {
                "valid": self.address_valid,
                "error": self.address_error,
            },
        }


# ──────────────────────────────────────────────────────────────
# Field validators
# ──────────────────────────────────────────────────────────────

def validate_national_id(nid: Optional[str]) -> tuple[bool, Optional[str]]:
    """
    Rules:
    1. Exactly 14 digits (Western numerals after normalisation)
    2. No letters
    3. Structural: valid century code (2 or 3), valid governorate code (01–35)
    """
    if not nid:
        return False, "National ID is missing"

    nid_norm = normalize_numerals(nid.strip())

    if not re.fullmatch(r"\d+", nid_norm):
        return False, f"Contains non-digit characters: {nid_norm!r}"

    if len(nid_norm) != 14:
        return False, f"Expected 14 digits, got {len(nid_norm)}"

    if not _validate_egyptian_id(nid_norm):
        return False, "Failed structural check (century code or governorate code invalid)"

    return True, None


def validate_arabic_name(name: Optional[str]) -> tuple[bool, Optional[str]]:
    """
    Rules:
    1. Non-empty
    2. Majority of alphabetic chars are Arabic Unicode
    3. Contains at least two words (first + last name minimum)
    4. No digits in the name
    """
    if not name or not name.strip():
        return False, "Name is missing"

    name = name.strip()

    if re.search(r"\d", name):
        return False, f"Name contains digits: {name!r}"

    arabic_chars = sum(1 for ch in name if "\u0600" <= ch <= "\u06FF")
    alpha_chars = sum(1 for ch in name if ch.isalpha())

    if alpha_chars == 0:
        return False, "Name has no alphabetic characters"

    ratio = arabic_chars / alpha_chars
    if ratio < 0.7:
        return False, f"Name appears non-Arabic (Arabic ratio: {ratio:.0%})"

    words = name.split()
    if len(words) < 2:
        return False, "Name should have at least two words"

    return True, None


def validate_address(address: Optional[str]) -> tuple[bool, Optional[str]]:
    """
    Light-touch validation:
    1. Non-empty
    2. Reasonable length (>= 5 characters)
    """
    if not address or not address.strip():
        return False, "Address is missing"
    if len(address.strip()) < 5:
        return False, "Address too short to be valid"
    return True, None


# ──────────────────────────────────────────────────────────────
# Text cleaner
# ──────────────────────────────────────────────────────────────

# Common OCR garbage patterns from ID card security backgrounds
_SECURITY_PATTERN_NOISE = re.compile(
    r"[<>|\\/*\^~`@#$%&_=+\[\]{};:\"!?]"
)


def clean_ocr_text(text: str) -> str:
    """
    Remove symbols introduced by the card's security/guilloche background,
    collapse extra whitespace, strip leading/trailing junk.
    """
    text = _SECURITY_PATTERN_NOISE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_result(result: OCRResult) -> OCRResult:
    """
    Apply cleaning to all text fields in an OCRResult in-place.
    """
    if result.name:
        result.name = _clean_arabic_text(clean_ocr_text(result.name))
    if result.address:
        result.address = _clean_arabic_text(clean_ocr_text(result.address))
    if result.national_id:
        # Strip any embedded spaces or dashes
        nid = normalize_numerals(result.national_id)
        result.national_id = re.sub(r"\D", "", nid)
    return result


# ──────────────────────────────────────────────────────────────
# Main post-processor
# ──────────────────────────────────────────────────────────────

class PostProcessor:
    """Cleans and validates an OCRResult, producing a final response dict."""

    def process(self, result: OCRResult) -> dict:
        """
        Args:
            result: Raw OCRResult from IDCardOCR.extract()

        Returns:
            {
                "data": { name, address, national_id, confidence, engine },
                "validation": ValidationReport.to_dict(),
                "success": bool,
            }
        """
        # 1. Clean
        cleaned = clean_result(result)

        # 2. Validate
        report = ValidationReport()

        nid_ok, nid_err = validate_national_id(cleaned.national_id)
        report.national_id_valid = nid_ok
        report.national_id_error = nid_err

        name_ok, name_err = validate_arabic_name(cleaned.name)
        report.name_valid = name_ok
        report.name_error = name_err

        addr_ok, addr_err = validate_address(cleaned.address)
        report.address_valid = addr_ok
        report.address_error = addr_err

        return {
            "data": cleaned.to_dict(),
            "validation": report.to_dict(),
            "success": report.all_valid,
        }
