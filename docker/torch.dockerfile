From pennylane/base:latest

# Update and install Torch
RUN apt-get update && pip3 install torch==1.8.1
# Image completed, Exit Now.
CMD echo "Successfully built Docker image."
