# Build from the repo root:
#   docker build -f baselines/trivial/Dockerfile -t celegans/trivial:latest .
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir numpy
COPY baselines/__init__.py /app/baselines/__init__.py
COPY baselines/common /app/baselines/common
COPY baselines/trivial/model.py baselines/trivial/predict.py baselines/trivial/predict.sh /app/baselines/trivial/
COPY baselines/trivial/model.pt /app/model.pt
RUN chmod +x /app/baselines/trivial/predict.sh
ENV PYTHONPATH=/app WEIGHTS=/app/model.pt
# Spec requires /predict.sh at image root.
RUN ln -s /app/baselines/trivial/predict.sh /predict.sh
CMD ["/predict.sh"]
