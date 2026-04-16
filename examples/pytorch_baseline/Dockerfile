# PyTorch + CUDA reference submission for the seg-in / seg-out contract.
#
# Base: pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
#   - Python 3.11 and torch 2.4.1 with CUDA 12.4 kernels already installed.
#   - No apt-get needed -> image builds reliably even on flaky networks
#     (archive.ubuntu.com is blocked/slow on some campus setups).
#   - Runs on GPU automatically when the worker passes --gpus all
#     (torch.cuda.is_available() -> True on the eval host).
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir numpy

WORKDIR /app
COPY predict.py /app/predict.py

CMD ["python", "/app/predict.py"]
