# CPU (ONNX-CPU)
FROM python:3.12-slim-bullseye AS focoos_apps-cpu
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
LABEL authors="focoos.ai"
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsm6 libxext6 gcc libmagic1 wget make cmake python3.12-tk && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Copy focoos-apps and install
COPY pyproject.toml ./pyproject.toml
COPY focoos_apps ./focoos_apps
RUN uv pip install --system -e .[onnx-cpu]


# GPU (ONNX GPU)
FROM ghcr.io/focoosai/deeplearning:base-cu12-cudnn9-py312-uv AS focoos-apps-gpu
LABEL authors="focoos.ai"
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends python3.12-tk && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Copy focoos-apps and install
COPY pyproject.toml ./pyproject.toml
COPY focoos_apps ./focoos_apps
RUN uv pip install --system -e .[onnx-gpu]


# TensorRT (ONNX GPU + TensorRT)
FROM ghcr.io/focoosai/deeplearning:cu12-cudnn9-py312-uv-tensorrt AS focoos-apps-tensorrt
LABEL authors="focoos.ai"
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends python3.12-tk && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Copy focoos-apps and install
COPY pyproject.toml ./pyproject.toml
COPY focoos_apps ./focoos_apps
RUN uv pip install --system -e .[tensorrt]



