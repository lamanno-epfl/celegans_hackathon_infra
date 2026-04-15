# Minimal image for the seg-in / seg-out contract.
# CPU-only baseline. For GPU, swap FROM to nvidia/cuda:12.4.1-runtime-ubuntu22.04
# and add `pip install torch --index-url https://download.pytorch.org/whl/cu124`.
FROM python:3.11-slim

RUN pip install --no-cache-dir numpy

WORKDIR /app
COPY predict.py /app/predict.py

CMD ["python", "/app/predict.py"]
