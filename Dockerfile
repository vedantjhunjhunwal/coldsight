# Runtime image: INT8 ONNX model + Tesseract, no PyTorch.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY coldsight coldsight
COPY models/coldsight_v2_int8.onnx models/
ENV COLDSIGHT_MODEL=/app/models/coldsight_v2_int8.onnx PORT=8000
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "-m", "coldsight.service"]
