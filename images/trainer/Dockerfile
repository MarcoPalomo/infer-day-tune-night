# Image du pipeline de nuit (et du bootstrap).
#   CPU / APU :  docker build -f images/trainer/Dockerfile .
#   CUDA      :  --build-arg BASE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --build-arg EXTRAS=train,cuda
#   ROCm      :  --build-arg BASE=rocm/pytorch:latest
ARG BASE=python:3.12-slim
FROM ${BASE}
ARG EXTRAS=train
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG LLAMACPP_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# Conversion LoRA PEFT -> GGUF pour le backend llama.cpp.
RUN git clone --depth 1 --branch ${LLAMACPP_REF} https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
 && pip install --no-cache-dir /opt/llama.cpp/gguf-py

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# Sur les images CUDA/ROCm, torch est déjà présent et n'est pas réinstallé.
RUN python3 -c "import torch" 2>/dev/null || pip install --no-cache-dir torch --index-url ${TORCH_INDEX}; \
    pip install --no-cache-dir ".[${EXTRAS}]"
ENV HF_HOME=/data/hf-cache
ENTRYPOINT ["idtn"]
