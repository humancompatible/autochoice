# Use Python 3.9 to match TensorFlow 2.20 and your pins
FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Helpful for rpy2 <-> R discovery, though usually auto-detected
    R_HOME=/usr/lib/R \
    RPY2_CFFI_MODE=ABI

# ---- System packages ----
# - r-base + r-base-dev for rpy2 (pulled by aif360[all])
# - libtirpc-dev fixes the exact link error you hit (-ltirpc)
# - cairo dev libs used by igraph[cairo]/cairocffi (pulled by [all])
# - toolchain + BLAS/LAPACK for scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    cmake \
    swig \
    pkg-config \
    r-base \
    r-base-dev \
    libtirpc-dev \
    libcairo2 \
    libcairo2-dev \
    libicu-dev \
    libpcre2-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libzstd-dev \
    libcurl4-openssl-dev \
    libgomp1 \
    libgfortran5 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


# ---- Python dependencies ----
# Keep your original pins; add pyarrow/hydra; install aif360[all] after system deps
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install \
      numpy==1.23.5 \
      scipy==1.9.3 \
      pandas==1.5.3 \
      scikit-learn==1.2.2 \
      pyarrow==15.0.2 \
      lightgbm==4.6.0 \
      "flaml[automl]" && \
    pip install 'aif360[all]==0.6.1' && \
    pip install \
      mlflow==2.22.0 \
      hyperopt \
      fairlearn \
      mapie==0.9.2 \
      hydra-core==1.3.2 \
      omegaconf==2.3.0 && \
    # CPU-only Torch; change to CUDA wheels if you’ll use GPU
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    # Optional extras you had
    pip install tensorflow==2.20.0 inFairness aix360


# ✅ Install AIX360 for explainability
RUN pip install --no-cache-dir aix360


RUN pip install --no-cache-dir \
    "git+https://github.com/humancompatible/explain.git@7406b59"

RUN pip install --no-cache-dir \
    "git+https://github.com/humancompatible/detect.git@93f8f32"


RUN pip install --no-cache-dir \
    "git+https://github.com/humancompatible/repair.git@3d08622a943df5f697bd0ec1b1f061ac99e1cdbf"

# Optional: create output dirs used by the repo's examples/plots
RUN mkdir -p /app/plots /app/data


WORKDIR /app

COPY dataset1M.parquet /data/
#COPY model.pkl /data/
COPY config.yaml .
COPY data_helper.py .
COPY run_mlflow.py .

ENTRYPOINT ["python", "run_mlflow.py"]

