FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    R_HOME=/usr/lib/R \
    RPY2_CFFI_MODE=ABI

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    libssl-dev \
    libffi-dev \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    libblas-dev \
    liblapack-dev \
    libstdc++6 \
    libpq-dev \
    libxrender1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgfortran5 \
    cmake \
    r-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy==1.23.5 scipy==1.9.3 pandas==1.5.3 scikit-learn==1.2.2

RUN pip install --no-cache-dir "flaml[automl]"
RUN pip install --no-cache-dir 'aif360[all]'
RUN pip install --no-cache-dir mlflow
RUN pip install --no-cache-dir hyperopt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fairlearn
RUN pip install --no-cache-dir tensorflow
RUN pip install --no-cache-dir inFairness
RUN pip install --no-cache-dir mapie

# ✅ Install AIX360 for explainability
RUN pip install --no-cache-dir aix360

WORKDIR /app

COPY data/datasetfile.csv /data/
COPY data/model.pkl /data/
COPY autochoice/autochoicebackend/run_mlflow.py .

ENTRYPOINT ["python", "run_mlflow.py"]

