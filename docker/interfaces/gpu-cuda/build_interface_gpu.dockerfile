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

FROM pennylane/cuda/base:latest
ARG INTERFACE_NAME=tensorflow
RUN . /opt/venv/bin/activate
# the following packages need re-installation in this build stage to avoid bugs
RUN pip uninstall -y numpy scipy rustworkx \
    && pip install "numpy<1.24" "scipy~=1.8" "rustworkx==0.12.0"
WORKDIR /opt/pennylane/docker/interfaces
RUN chmod +x install-interface-gpu.sh && ./install-interface-gpu.sh $INTERFACE_NAME

# Run Unit-Tests again
WORKDIR /opt/pennylane
RUN make test
# Image build completed.
CMD echo "Successfully built Docker image"
