# 1. CUDA-ready PyTorch base image
# FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04
# FROM vllm/vllm-openai:latest


# 2. Set working directory
WORKDIR /app

# 3. Install required packages (Flask + Git)
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
# RUN apt-get install python-pip 
# RUN pip3 install -v flask
# 4. Clone the required repo (at build time)
#    Or comment this out to do it dynamically at runtime instead
RUN git clone https://github.com/whisperzh/MG_test.git

COPY  Ipchandler_demo /app/MG_test/megatron/expert_replicate/Ipchandler_demo
COPY ./Ipchandler_demo/tensor_utils_pybind.py /app/MG_test/megatron/core/transformer/moe/


# RUN cp /app/MG_test/megatron/expert_replicate/Ipchandler\ demo/server.ipynb /app/MG_test/megatron/core/transformer/moe/

RUN cd MG_test && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    pip install  -e .  && \
    pip install  transformers  && \
    pip install  accelerate  && \
    pip install  --no-build-isolation transformer_engine[pytorch]


RUN bash -c "cd MG_test && \
    source venv/bin/activate && \
    cd megatron/expert_replicate/Ipchandler_demo/install_me_first && \
    sh build.sh"

COPY ./expert_container.py /app/MG_test/megatron/expert_replicate/
COPY ./tensor_utils_pybind.py /app/MG_test/megatron/expert_replicate/
# 设置工作目录为MG_test（方便后续操作）
WORKDIR /app/MG_test
ENV PYTHONPATH="/app/MG_test:$PYTHONPATH"
CMD ["bash", "-c", "source venv/bin/activate && python3 -m megatron.expert_replicate.expert_container"]
