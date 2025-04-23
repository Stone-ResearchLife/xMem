# Use an official Ubuntu 20.04 base image with CUDA 11.7 support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY ColossalAI /tmp/ColossalAI
RUN cd /tmp/ColossalAI && \
    pip install . && \
    rm -rf /tmp/ColossalAI

# Install other dependencies
RUN pip install \
    transformers==4.29.2 \
    GPUtil \
    pynvml \
    sentencepiece \
    datasets \
    deepspeed \
    sacremoses \
    openai==0.27.9 && \
    pip install --upgrade accelerate \
