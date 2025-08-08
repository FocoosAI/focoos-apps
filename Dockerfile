# CPU / ONNX-CPU
FROM python:3.12-slim-bullseye AS focoos-cpu
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
LABEL authors="focoos.ai"
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsm6 libxext6 gcc libmagic1 wget make cmake && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install the Focoos SDK from github with ONNX-CPU extras
RUN uv pip install --system 'focoos[onnx-cpu] @ git+https://github.com/FocoosAI/focoos.git'

# Copy focoos-apps code
COPY pyproject.toml ./pyproject.toml
COPY README.md ./README.md
COPY focoos_apps ./focoos_apps
# Install focoos-apps
# RUN uv pip install --system -e .

# Thin runtime image for ONNX-CPU
FROM focoos-cpu AS focoos-apps-cpu
ENTRYPOINT ["focoos-apps"]
CMD ["--help"]


# GPU base (PyTorch / ONNX GPU)
FROM ghcr.io/focoosai/deeplearning:base-cu12-cudnn9-py312-uv AS focoos-gpu
LABEL authors="focoos.ai"
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install the Focoos SDK from github with ONNX-GPU extras
RUN uv pip install --system 'focoos[onnx] @ git+https://github.com/FocoosAI/focoos.git'

# Copy focoos-apps and install
COPY pyproject.toml ./pyproject.toml
COPY README.md ./README.md
COPY focoos_apps ./focoos_apps
# RUN uv pip install --system -e .

FROM focoos-gpu AS focoos-apps-gpu
ENTRYPOINT ["focoos-apps"]
CMD ["--help"]


# TensorRT variant
FROM focoos-gpu AS focoos-tensorrt
RUN uv pip install --system 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git'

FROM focoos-tensorrt AS focoos-apps-tensorrt
ENTRYPOINT ["focoos-apps"]
CMD ["--help"]
