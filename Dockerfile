# 3DGS + cycle-diffusion (DDPM/DDIM) style-baking pipeline.
#
# Built for CUDA 12.1 / recent Ada+ GPUs (e.g. RTX 4090, sm_89). CUDA 11.8's nvcc
# fails to compile diff-gaussian-rasterization against PyTorch >= 2.1's generated
# ATen headers (c10::optional parse error in ATen/ops/*_ops.h under nvcc), so this
# needs 12.1+. TORCH_CUDA_ARCH_LIST is narrowed to sm_89 (RTX 4090) only; widen it
# (e.g. back to "7.0;7.5;8.0;8.6;8.9") to target other GPU generations too, at the
# cost of more build time/memory (nvcc compiles each .cu file once per arch).
#
# MAX_JOBS caps ninja's build parallelism: each nvcc job compiling these CUDA
# extensions can use several GB of RAM, and ninja defaults to one job per CPU
# core, so on a many-core/limited-RAM box the unbounded default reliably OOMs
# mid-build (surfaces as nvcc/cicc segfaults or unrelated-looking Python crashes
# in whichever process the kernel kills). Raise it if you have more RAM per core.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    MAX_JOBS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git wget ninja-build build-essential \
        colmap ffmpeg \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel ninja

RUN python3 -m pip install --no-cache-dir \
        torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

# numpy<2: the torch wheel above is built against the NumPy 1.x ABI.
RUN python3 -m pip install --no-cache-dir \
        "numpy<2" pillow pyyaml tqdm plyfile joblib opencv-python-headless

WORKDIR /workspace
COPY . /workspace

# Build the CUDA rasterizer/kNN/SSIM extensions into site-packages (not editable),
# so they stay intact even when ./input, ./output, and cycle-diffusion-for-anime-face/ckpts
# are bind-mounted over the corresponding paths at runtime.
# --no-build-isolation: their setup.py imports torch directly at build time, but
# doesn't declare it as a PEP517 build requirement, so pip's isolated build venv
# wouldn't see the torch we just installed above.
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
        ./submodules/diff-gaussian-rasterization \
        ./submodules/simple-knn \
        ./submodules/fused-ssim

ENTRYPOINT ["python3", "style_transfer.py"]
CMD ["--help"]
