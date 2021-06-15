From pennylane:latest

# Update and install Torch
<<<<<<< HEAD
RUN apt-get update && pip install torch==1.8.1
=======
RUN apt-get update && pip3 install torch==1.8.1
>>>>>>> 9609b7777edf1eab44d038ec58e1338fb232378f
# Image completed, Exit Now.
CMD echo "Docker image Builing process is Successful"
