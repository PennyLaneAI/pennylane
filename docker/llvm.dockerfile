# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM pennylane/base:latest AS data-image

FROM ubuntu:latest AS compile-image

# Setup and install Basic packages
RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y apt-utils --no-install-recommends
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata \
    build-essential \
    ca-certificates \
    ccache \
    clang \
    lld \
    git \
    ninja-build \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/sbin/update-ccache-symlinks \
    && mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache \
    && python3 -m venv /opt/venv

# Activate VirtualENV
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /opt/llvm-build

# Copy install script
COPY --from=data-image /opt/pennylane/docker/install-llvm-mlir.sh .

# Build LLVM with MLIR
ENV VERSION=14.0.0
RUN chmod +x install-llvm-mlir.sh && ./install-llvm-mlir.sh

# Create Second small(er) build.
FROM pennylane/base:latest
COPY --from=compile-image /opt/llvm /opt/llvm
ENV VERSION=14.0.0
COPY --from=compile-image /opt/llvm-build/llvm-project-${VERSION}.src/mlir/python/requirements.txt /opt/llvm
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -r /opt/llvm/requirements.txt
ENV PYTHONPATH=/opt/llvm/python_packages/mlir_core

WORKDIR /opt/pennylane

RUN python3 -c 'import pennylane as qml; from mlir.ir import Context, Module; from mlir.dialects import builtin;'

# Image build completed.
CMD echo "Successfully built LLVM/MLIR image"
