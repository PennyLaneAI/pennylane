From pennylane:latest

# Update and install Torch
RUN apt-get update && pip install torch==1.8.1
# Image completed, Exit Now.
CMD echo "Docker image Builing process is Successful"
