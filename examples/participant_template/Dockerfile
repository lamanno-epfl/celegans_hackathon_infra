# Build from repo root: docker build -f examples/participant_template/Dockerfile -t myteam/model:v1 .
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY examples/participant_template/predict.py examples/participant_template/predict.sh /app/
RUN chmod +x /app/predict.sh && ln -s /app/predict.sh /predict.sh
CMD ["/predict.sh"]
