FROM ubuntu:20.04 AS compile-image

# Setup and install Basic packages
RUN apt-get update && apt-get install -y apt-utils --no-install-recommends
RUN DEBIAN_FRONTEND="noninteractive" apt-get install tzdata
RUN apt-get install -y build-essential \
        tzdata \
        ca-certificates \
        ccache \
        cmake \
        curl \
	      git \
	      python3 \
        python3-pip \
        python3-venv \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

# Create and activate VirtualENV
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

#Setup and Build pennylane
WORKDIR /opt/pennylane
COPY  . .
RUN git submodule update --init --recursive
RUN  pip install wheel && pip install -r requirements.txt \
        && python3 setup.py install \
        && pip install pytest pytest-cov pytest-mock flaky \
        && make test

# create Second small build.
FROM ubuntu:20.04
COPY --from=compile-image /opt/venv /opt/venv
# Get PennyLane Source to use for Unit-test at later stage
COPY --from=compile-image /opt/pennylane /opt/pennylane
ENV PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y apt-utils --no-install-recommends python3 python3-pip python3-venv
# Image completed, Exit Now.
CMD echo "Successfully built Docker image"
