FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy scipy
COPY examples/blur_participant/predict.py examples/blur_participant/predict.sh /app/
RUN chmod +x /app/predict.sh && ln -s /app/predict.sh /predict.sh
CMD ["/predict.sh"]
