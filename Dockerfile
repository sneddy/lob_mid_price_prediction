FROM pytorch/pytorch
# 04 defining project root env
ENV PROJECT_ROOT /xtx
WORKDIR ${PROJECT_ROOT}

# gcc
RUN apt-get update && \
    apt-get -y install build-essential git gcc python-dev libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ${PROJECT_ROOT}/requirements.txt
RUN pip install -r ${PROJECT_ROOT}/requirements.txt

# install fastFm
# RUN git clone --recursive https://github.com/ibayer/fastFM.git /workspace/fastFM
# RUN cd /workspace/fastFM && pip install -r ./requirements.txt
# RUN cd /workspace/fastFM PYTHON=python3 make 
# RUN pip install .

