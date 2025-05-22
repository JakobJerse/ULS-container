FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base

# Remove old CUDA repo, add deadsnakes, install Python 3.10 and build deps
RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      git wget unzip libopenblas-dev \
      python3.10 python3.10-dev python3.10-venv python3.10-distutils \
      nano \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Bootstrap a clean pip under Python 3.10
RUN python3.10 -m ensurepip --upgrade && \
    python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install your frozen requirements
COPY requirements.txt /tmp/requirements.txt
RUN python3.10 -m pip install --no-cache-dir \
      -r /tmp/requirements.txt \
      -f https://download.pytorch.org/whl/torch_stable.html

# Clone nnU-Net at your desired commit or tag
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet && \
    cd /opt/algorithm/nnunet && \
    # git checkout 947eafbb9adb5eb06b9171330b4688e006e6f301

# Install nnU-Net in editable mode plus extras
RUN python3.10 -m pip install --no-cache-dir \
      -e /opt/algorithm/nnunet \
      graphviz onnx SimpleITK && \
    rm -rf /home/user/.cache/pip

### USER SETUP
RUN groupadd -r user && useradd -m -r -g user user && \
    chown -R user:user /opt/algorithm

RUN mkdir -p /opt/app /input /output && \
    chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

# Copy your inference scripts
COPY --chown=user:user process.py export2onnx.py /opt/app/

# (Optional) custom trainers/extensions
COPY --chown=user:user ./architecture/extensions/nnunetv2/ /opt/algorithm/nnunet/nnunetv2/

# Set nnU-Net env vars
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT ["python3.10", "-m", "process"]
