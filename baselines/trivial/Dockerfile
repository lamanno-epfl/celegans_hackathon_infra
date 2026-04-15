FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy torch torchvision
COPY model.py predict.py predict.sh ./
COPY ../common/rotation6d.py rotation6d.py
RUN chmod +x /app/predict.sh
ENV PYTHONPATH=/app
CMD ["/app/predict.sh"]
