# Build from repo root: docker build -f my_sub/Dockerfile -t myteam/model:v1 .
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY my_sub/predict.py my_sub/predict.sh /app/
RUN chmod +x /app/predict.sh && ln -s /app/predict.sh /predict.sh
CMD ["/predict.sh"]
