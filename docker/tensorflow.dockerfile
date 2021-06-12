From pennylane:latest

# Update and install TensorFlow
RUN apt-get update && pip3 install tensorflow==2.5.0
# Image completed, Exit Now.
CMD echo "Docker image Builing process is Successful"
