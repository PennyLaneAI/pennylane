From pennylane/cuda/base:latest

ARG INTERFACE_NAME=tensorflow
# Build Jax interface
RUN if [ "$INTERFACE_NAME" = "jax" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install jax==0.2.14 \
    jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html;fi

# Build TensorFlow interface
RUN if [ "$INTERFACE_NAME" = "tensorflow" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install tensorflow==2.5.0;fi

# Build Torch interface
RUN if [ "$INTERFACE_NAME" = "torch" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install torch==1.8.1;fi

# Build All interfaces together
RUN if [ "$INTERFACE_NAME" = "all" ] ; then \
    apt-get update && apt-get -y install --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* && pip3 install tensorflow==2.5.0 torch==1.8.1 jax==0.2.14 \
    jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html;fi

# Run Unit-Tests again
WORKDIR /opt/pennylane
RUN make test
# Image completed, Exit Now.
CMD echo "Successfully built Docker image"
