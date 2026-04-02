FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
 
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /workspace