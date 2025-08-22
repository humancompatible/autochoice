FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Helpful for rpy2 <-> R discovery, though usually auto-detected
    R_HOME=/usr/lib/R \
    RPY2_CFFI_MODE=ABI


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
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install tensorflow==2.20.0 inFairness aix360


# ✅ Install AIX360 for explainability
RUN pip install --no-cache-dir aix360

WORKDIR /app

COPY dataset1M.parquet /data/
#COPY model.pkl /data/
COPY data_helper.py .
COPY run_mlflow.py .

ENTRYPOINT ["python", "run_mlflow.py"]

