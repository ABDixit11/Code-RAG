FROM ubuntu:20.04

ARG ROCKSDB_VERSION=v9.2.1
ARG PYTHON_VERSION=3.9.20

# Set non-interactive to prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_PYTHON_REFRESH=quiet

# Update and install required packages
RUN apt-get update --fix-missing && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libffi-dev \
    libssl-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    liblz4-dev \
    libsnappy-dev \
    libzstd-dev \
    libblas-dev \
    liblapack-dev \
    perl \
    g++ \
    nlohmann-json3-dev && \
    echo "Base dependencies installed"

# Download and compile Python 3.9 from source
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -sf /usr/local/bin/python3.9 /usr/bin/python3 && \
    python3 --version && \
    echo "Python 3.9 compiled and installed"

# Install RocksDB dependencies (Original Commands)
RUN [ ! -d /usr/src ] && mkdir /usr/src || echo "/usr/src already exists" && \
    cd /usr/src && \
    git clone --depth 1 --branch ${ROCKSDB_VERSION} https://github.com/facebook/rocksdb.git && \
    cd /usr/src/rocksdb && \
    echo "RocksDB cloned successfully" && \
    sed -i 's/install -C/install -c/g' Makefile && \
    make -j4 shared_lib && \
    make install-shared && \
    echo "RocksDB built and installed"

# Clean up unnecessary packages and source files
RUN apt-get remove --purge -y perl wget && \
    apt-get autoremove -y && \
    rm -rf /usr/src/Python-${PYTHON_VERSION} /usr/src/Python-${PYTHON_VERSION}.tgz /usr/src/rocksdb && \
    echo "Cleanup completed"

# Set up Python virtual environment and install packages
RUN python3 -m venv /env && \
    /env/bin/pip install --upgrade pip && \
    /env/bin/pip install datasets pandas numpy scipy cmake faiss-cpu sentence_transformers && \
    /env/bin/pip install bm25s Pystemmer jax[cpu] ragatouille accelerate && \
    echo "Python environment and packages installed"

# Set the working directory
WORKDIR /app

# Copy source files into the container
COPY get_data.py coderag.py load_data batch_fetch vector_storage.py setup.sh /app/

# Ensure the virtual environment is activated by default
ENV VIRTUAL_ENV=/env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Give execute permissions to setup.sh
RUN chmod +x setup.sh

# Start the container by running coderag.py
CMD ["tail", "-f", "/dev/null"]
