FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY predict.py predict.sh ./
RUN chmod +x /app/predict.sh
CMD ["/app/predict.sh"]
