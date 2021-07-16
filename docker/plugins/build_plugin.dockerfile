<<<<<<< HEAD
# Copyright 2018-2021 Xanadu Quantum Technologies Inc.
=======
From pennylane/base:latest
ARG PLUGIN_NAME=qiskit

# Build Qiskit Plugin
RUN if [ "$PLUGIN_NAME" = "qiskit" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-qiskit;fi

# Build Amazon-Braket Plugin
RUN if [ "$PLUGIN_NAME" = "amazon-braket" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install amazon-braket-pennylane-plugin;fi

# Build SF Plugin
RUN if [ "$PLUGIN_NAME" = "sf" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-sf;fi

# Build Cirq Plugin
RUN if [ "$PLUGIN_NAME" = "cirq" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-cirq;fi

# Build Qulacs Plugin
RUN if [ "$PLUGIN_NAME" = "qulacs" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install qulacs pennylane-qulacs;fi
>>>>>>> 7d2d7942655b1118b4234b9278440aed2174465c

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

<<<<<<< HEAD
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
=======
# Build PQ Plugin
RUN if [ "$PLUGIN_NAME" = "pq" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-pq;fi

# Build Q# Plugin
RUN if [ "$PLUGIN_NAME" = "qsharp" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-qsharp;fi

# Build Forest Plugin
RUN if [ "$PLUGIN_NAME" = "forest" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-forest;fi

# Build Orquestra Plugin
RUN if [ "$PLUGIN_NAME" = "orquestra" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-orquestra;fi

# Build Ionq Plugin
RUN if [ "$PLUGIN_NAME" = "ionq" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-ionq;fi
>>>>>>> 7d2d7942655b1118b4234b9278440aed2174465c

From pennylane/base:latest
ARG PLUGIN_NAME=tensorflow
WORKDIR /opt/pennylane/docker
RUN chmod +x install.sh && ./install-plugin.sh $PLUGIN_NAME
# Run Unit-Tests again
WORKDIR /opt/pennylane
RUN make test
<<<<<<< HEAD
# Image completed, Exit Now.
CMD echo "Successfully built Docker image!"
=======
# Image build completed.
CMD echo "Successfully built Docker image"
>>>>>>> 7d2d7942655b1118b4234b9278440aed2174465c
