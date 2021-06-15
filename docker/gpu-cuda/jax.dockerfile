From pennylane/cuda/base:latest

# Update and install Jax
RUN apt-get update && pip3 install jax jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# Image completed, Exit Now.
CMD echo "Docker image Builing process is Successful"
