From pennylane/cuda/base:latest

# Update and install Jax
RUN apt-get update && apt-get -y install make  \
    && rm -rf /var/lib/apt/lists/* && pip3 install jax  \
    jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
    
# Run Unit-Tests again
WORKDIR /opt/pennylane
RUN make test
# Image completed, Exit Now.
CMD echo "Successfully built Docker image"
