From pennylane/base:latest

# Update and install Jax
RUN apt-get update && apt-get -y install --no-install-recommends make && rm -rf /var/lib/apt/lists/* && pip3 install jax jaxlib
# Run Unit-Tests again
WORKDIR /opt/pennylane
RUN make test
# Image completed, Exit Now.
CMD echo "Successfully built Docker image"
