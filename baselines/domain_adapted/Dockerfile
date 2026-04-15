# Build from the repo root:
#   docker build -f baselines/domain_adapted/Dockerfile -t celegans/domain_adapted:latest .
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir numpy
COPY baselines/__init__.py /app/baselines/__init__.py
COPY baselines/common /app/baselines/common
COPY baselines/domain_adapted/model.py baselines/domain_adapted/predict.py baselines/domain_adapted/predict.sh /app/baselines/domain_adapted/
COPY baselines/domain_adapted/model.pt /app/model.pt
RUN chmod +x /app/baselines/domain_adapted/predict.sh
ENV PYTHONPATH=/app WEIGHTS=/app/model.pt
RUN ln -s /app/baselines/domain_adapted/predict.sh /predict.sh
CMD ["/predict.sh"]
