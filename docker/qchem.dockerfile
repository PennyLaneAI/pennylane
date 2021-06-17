From pennylane/base:latest

# Update and install Qchem
RUN apt-get update && apt-get install -y make git
# Create and activate VirtualENV
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"*" "

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
