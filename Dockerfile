FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base

# Remove old CUDA apt source
RUN rm /etc/apt/sources.list.d/cuda.list

# 1) Install tooling to add PPAs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 2) Add deadsnakes PPA and install Python 3.10 and essentials
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    libopenblas-dev \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap and upgrade pip under Python 3.10
RUN python3.10 -m ensurepip --upgrade && \
    python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies from your frozen requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN python3.10 -m pip install --no-cache-dir \
    -r /tmp/requirements.txt \
    -f https://download.pytorch.org/whl/torch_stable.html

# Configure Git and clone nnU-Net (latest commit of default branch)
RUN git config --global advice.detachedHead false && \
    git clone --depth 1 https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet

# Install nnU-Net in editable mode plus extras
RUN python3.10 -m pip install --no-cache-dir \
    -e /opt/algorithm/nnunet \
    graphviz \
    onnx \
    SimpleITK \
    && rm -rf /home/user/.cache/pip

### USER SETUP
RUN groupadd -r user && \
    useradd -m --no-log-init -r -g user user && \
    chown -R user:user /opt/algorithm && \
    mkdir -p /opt/app /input /output && \
    chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

# Copy inference scripts
COPY --chown=user:user process.py export2onnx.py /opt/app/

### OPTIONAL: Custom nnU-Net extensions (uncomment if needed)
# COPY --chown=user:user ./architecture/extensions/nnunetv2/ /opt/algorithm/nnunet/nnunetv2/

# Environment variables for nnU-Net data paths
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT ["python3.10", "-m", "process"]
