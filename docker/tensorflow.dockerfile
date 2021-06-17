From pennylane/base:latest

# Update and install TensorFlow
RUN apt-get update && apt-get -y install --no-install-recommends make &&  rm -rf /var/lib/apt/lists/*  && pip3 install tensorflow==2.5.0
# Run Unit-Tests again
WORKDIR /opt/pennylane
RUN make test
# Image completed, Exit Now.
CMD echo "Successfully built Docker image."
