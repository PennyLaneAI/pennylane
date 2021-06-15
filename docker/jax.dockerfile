From pennylane:latest

# Update and install Jax
RUN apt-get update && pip3 install jax jaxlib
# Image completed, Exit Now.
CMD echo "Docker image Builing process is Successful"
