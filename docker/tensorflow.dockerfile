From pennylane:latest

# Update and install TensorFlow
<<<<<<< HEAD
RUN apt-get update && pip install tensorflow==2.5.0
=======
RUN apt-get update && pip3 install tensorflow==2.5.0
>>>>>>> 9609b7777edf1eab44d038ec58e1338fb232378f
# Image completed, Exit Now.
CMD echo "Docker image Builing process is Successful"
