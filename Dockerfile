FROM python:3.10 AS base

RUN apt-get update -y && apt-get install build-essential -y
RUN pip install h5py zstd dill 'numpy<1.24'

WORKDIR /usr/src/pennylane

COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY Makefile Makefile

COPY pennylane/_version.py pennylane/_version.py

RUN make install

COPY pennylane pennylane
RUN pip install -e .
