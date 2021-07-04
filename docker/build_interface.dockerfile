From pennylane/base:latest
ARG INTERFACE_NAME=tensorflow

RUN if [ "$INTERFACE_NAME" = "jax" ] ; then \
       apt-get update && apt-get -y install --no-install-recommends make \
       && rm -rf /var/lib/apt/lists/* && pip3 install jax jaxlib;fi

RUN if [ "$INTERFACE_NAME" = "tensorflow" ] ; then \
      apt-get update && apt-get -y install --no-install-recommends make \
      && rm -rf /var/lib/apt/lists/* && pip3 install tensorflow==2.5.0;fi

RUN if [ "$INTERFACE_NAME" = "tensorflow" ] ; then \
      apt-get update && apt-get -y install --no-install-recommends make \
      && rm -rf /var/lib/apt/lists/* && pip3 install torch==1.8.1;fi

# Run Unit-Tests again
WORKDIR /opt/pennylane
RUN make test
# Image completed, Exit Now.
CMD echo "Successfully built Docker image $INTERFACE_NAME"
