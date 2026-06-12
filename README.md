# 🪪 Egyptian National ID — OCR Pipeline

An end-to-end Optical Character Recognition pipeline for **Egyptian National ID cards**.
Given a photo of an ID card (possibly tilted, poorly lit, or low-resolution), the
pipeline returns the cardholder's **Full Name**, **Address**, and **14-digit National ID Number**
as clean, validated JSON.

```json
{
  "data": {
    "name": "أحمد محمد علي حسن",
    "address": "القاهرة شارع التحرير مدينة نصر",
    "national_id": "29005010123456",
    "confidence": 0.93,
    "engine": "easyocr"
  },
  "validation": {
    "all_valid": true,
    "national_id": { "valid": true, "error": null },
    "name":        { "valid": true, "error": null },
    "address":     { "valid": true, "error": null }
  },
  "success": true,
  "processing_time_ms": 842.3,
  "warp_applied": true
}
