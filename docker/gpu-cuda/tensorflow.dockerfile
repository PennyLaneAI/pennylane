From pennylane/cuda/base:latest

# Update and install TensorFlow
RUN apt-get update && apt-get install make &&  pip3 install tensorflow==2.5.0
# Run Unit-Tests again
WORKDIR /opt/pennylane
RUN make test
# Image completed, Exit Now.
CMD echo "Successfully built Docker image"
