# ─────────────────────────────────────────────────────────
# Egyptian National ID OCR — API Container
# ─────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps required by OpenCV and EasyOCR (libGL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download EasyOCR models at build time (optional, avoids cold-start delay)
# RUN python -c "import easyocr; easyocr.Reader(['ar','en'])"

ENV OCR_ENGINE=easyocr
ENV USE_GPU=0
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
