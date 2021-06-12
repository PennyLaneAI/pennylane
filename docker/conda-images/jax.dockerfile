
ARG BASE_IMAGE=ubuntu:18.04
ARG PYTHON_VERSION=3.7
# Setup develop base image packages(build-essentials etc)
FROM ${BASE_IMAGE} as dev-base

RUN apt-get update && apt-get install -y --no-install-recommends
RUN DEBIAN_FRONTEND="noninteractive" apt-get install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

# Setup Miniconda
FROM dev-base as conda
ARG PYTHON_VERSION=3.7
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# Setup as Submodule-intermediate image for pennylane
FROM dev-base as submodule-update
COPY --from=conda /opt/conda /opt/conda
WORKDIR /opt/pennylane
COPY  . .
RUN git submodule update --init --recursive
RUN conda create -q -n docker-environment python=${PYTHON_VERSION} -y \
        && conda init bash  \
        && . /root/.bashrc \
        && conda update conda  \
        && conda activate docker-environment \
        && pip install -r requirements.txt \
        && python setup.py install \
        && pip install jax jaxlib \
        && pip install pytest pytest-cov pytest-mock flaky \
        && make test
