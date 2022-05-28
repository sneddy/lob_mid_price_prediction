FROM pytorch/pytorch
# 04 defining project root env
ENV PROJECT_ROOT /xtx
WORKDIR ${PROJECT_ROOT}

# gcc
RUN apt-get update && \
    apt-get -y install gcc git python-dev libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# install fastFm
RUN git clone --recursive https://github.com/ibayer/fastFM.git /workspace && \
    cd /workspace/fastFM && \
    pip install -r ./requirements.txt && \
    PYTHON=python3 make && \
    pip install .

COPY requirements.txt ${PROJECT_ROOT}/requirements.txt
RUN pip install -r ${PROJECT_ROOT}/requirements.txt