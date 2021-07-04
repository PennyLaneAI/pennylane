From pennylane/base:latest
ARG INTERFACE_NAME #= jax jaxlib tensorflow==2.5.0 torch==1.8.1

# need to add plugin support from Make-File

# Update and install Jax
RUN apt-get update && apt-get -y install --no-install-recommends make  \
    && rm -rf /var/lib/apt/lists/* && pip3 install $INTERFACE_NAME
# Run Unit-Tests again
WORKDIR /opt/pennylane
RUN make test
# Image completed, Exit Now.
CMD echo "Successfully built Docker image"
