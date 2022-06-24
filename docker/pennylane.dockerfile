# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:latest AS compile-image

# Setup and install Basic packages
RUN apt-get update && apt-get install -y apt-utils --no-install-recommends
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata \
    build-essential \
    tzdata \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/* \
    && /usr/sbin/update-ccache-symlinks \
    && mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache \
    && python3 -m venv /opt/venv
# Activate VirtualENV
ENV PATH="/opt/venv/bin:$PATH"

#Setup and Build pennylane
WORKDIR /opt/pennylane
COPY  . .
RUN git submodule update --init --recursive
RUN pip install wheel && pip install -r requirements.txt
RUN python3 setup.py install
RUN pip install pytest pytest-cov pytest-mock flaky
RUN pip install -i https://test.pypi.org/simple/ pennylane-lightning --pre --upgrade
# hotfix, remove when pyscf 2.1 is released (currently no wheel for python3.10)
RUN pip install openfermionpyscf || true
RUN make test && make coverage

# create Second small build.
FROM ubuntu:latest
COPY --from=compile-image /opt/venv /opt/venv
# Get PennyLane Source to use for Unit-tests at later stage
COPY --from=compile-image /opt/pennylane /opt/pennylane
ENV PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y apt-utils \
    --no-install-recommends python3 python3-pip python3-venv
# Image build completed.
CMD echo "Successfully built Docker image"
