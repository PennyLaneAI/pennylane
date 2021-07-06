From pennylane/base:latest
ARG PLUGIN_NAME=tensorflow

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
    && rm -rf /var/lib/apt/lists/* && $ pip3 install pennylane-cirq;fi

# Build Qulacs Plugin
RUN if [ "$PLUGIN_NAME" = "qulacs" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install qulacs pennylane-qulacs;fi

# Build AQT Plugin
RUN if [ "$PLUGIN_NAME" = "aqt" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-aqt;fi

# Build Honeywell Plugin
RUN if [ "$PLUGIN_NAME" = "honeywell" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-honeywell;fi

# Build PQ Plugin
RUN if [ "$PLUGIN_NAME" = "pq" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane_pq;fi

# Build Qsharp Plugin
RUN if [ "$PLUGIN_NAME" = "qsharp" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-qsharp;fi

# Build Forest Plugin
RUN if [ "$PLUGIN_NAME" = "forest" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-forest;fi

# Build orquestra Plugin
RUN if [ "$PLUGIN_NAME" = "orquestra" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-orquestra;fi

# Build Ionq Plugin
RUN if [ "$PLUGIN_NAME" = "ionq" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-ionq;fi

# Build All Plugins together
RUN if [ "$PLUGIN_NAME" = "all" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install pennylane-qiskit \
    amazon-braket-pennylane-plugin \
    pennylane-sf \
    pennylane-cirq \
    qulacs pennylane-qulacs \
    pennylane-aqt \
    pennylane-honeywell \
    pennylane_pq \
    pennylane-qsharp \
    pennylane-forest \
    pennylane-orquestra \
    pennylane-ionq;fi

# Run Unit-Tests again
WORKDIR /opt/pennylane
RUN make test
# Image completed, Exit Now.
CMD echo "Successfully built Docker image!"
