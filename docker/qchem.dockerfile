From pennylane/base:latest

# Update and install Qchem
RUN DEBIAN_FRONTEND="noninteractive" apt-get install tzdata # need to perform this again
RUN apt-get update && apt-get -y install --no-install-recommends make git \
    &&  rm -rf /var/lib/apt/lists/*

# Setup and Build Qchem
WORKDIR /opt/pennylane/qchem
COPY  . .
RUN git submodule update --init --recursive
RUN  pip install wheel && pip install openfermionpyscf && pip install -r requirements.txt \
        && python3 setup.py install \
        && make test

# Image completed, Exit Now.
CMD echo "Successfully built Docker image"
