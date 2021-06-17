From pennylane/base:latest

# Update and install Jax
RUN apt-get update
# Create and activate VirtualENV
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Setup and Build Qchem
WORKDIR /opt/pennylane/qchem
COPY  . .
RUN git submodule update --init --recursive
RUN  pip install wheel && pip install openfermionpyscf && pip install -r requirements.txt \
        && python3 setup.py install \
        && pip install pytest pytest-cov pytest-mock flaky \
        && make test

# Image completed, Exit Now.
CMD echo "Successfully built Docker image"

## Left for some Experiments on More Concise Build
# create Second small build.
##FROM ubuntu:20.04
##COPY --from=compile-image /opt/venv /opt/venv
##ENV PATH="/opt/venv/bin:$PATH"
##RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv
# Image completed, Exit Now.
##CMD echo "Docker image Builing process is Successful"
