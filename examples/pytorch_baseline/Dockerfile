# PyTorch reference submission for the seg-in / seg-out contract.
# CPU+GPU capable. Built on the CUDA runtime image; torch will auto-use GPU if
# `--gpus all` is passed (the worker does this when ENABLE_GPU=1 on the host).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels — smaller image, runs on any host. Swap the URL to the cu124 wheel
# if you want to ship GPU kernels inside the image (adds ~2 GB).
RUN pip3 install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1

RUN pip3 install --no-cache-dir numpy

WORKDIR /app
COPY predict.py /app/predict.py

CMD ["python3", "/app/predict.py"]
