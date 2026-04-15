# Build from the repo root:
#   docker build -f baselines/degenerate/Dockerfile -t celegans/degenerate:latest .
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY baselines/degenerate/predict.py baselines/degenerate/predict.sh /app/
RUN chmod +x /app/predict.sh
RUN ln -s /app/predict.sh /predict.sh
CMD ["/predict.sh"]
