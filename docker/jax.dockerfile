From pennylane:latest

# Update and install Jax
<<<<<<< HEAD
RUN apt-get update && pip install jax jaxlib
=======
RUN apt-get update && pip3 install jax jaxlib
>>>>>>> 9609b7777edf1eab44d038ec58e1338fb232378f
# Image completed, Exit Now.
CMD echo "Docker image Builing process is Successful"
