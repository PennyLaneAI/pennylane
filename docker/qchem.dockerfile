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

FROM pennylane/base:latest

# Update and install Qchem
RUN DEBIAN_FRONTEND="noninteractive" apt-get install tzdata # need to perform this again
RUN apt-get update \
    && rm -rf /var/lib/apt/lists/*

# Setup and Build Qchem
WORKDIR /opt/pennylane/qchem
RUN git submodule update --init --recursive
RUN pip install wheel && pip install openfermionpyscf && pip install -r requirements.txt \
    && python3 setup.py install \
    && pip install -i https://test.pypi.org/simple/ pennylane-lightning --pre --upgrade \
    && make test && make coverage

# Image build completed.
CMD echo "Successfully built Docker image"
