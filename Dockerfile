# Multi-stage Docker setup for empirical-bayes replication package
# Handles Python 3.9, R 4.4, rpy2, renv, and MOSEK integration

FROM rocker/r-ver:4.4.0

# Set environment variables for reproducibility
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Add deadsnakes PPA for Python 3.10
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install system dependencies for Python, R, and MOSEK
RUN apt-get update && apt-get install -y \
    # Python and pip
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    # System libraries needed by Python packages
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    # Libraries for geospatial packages
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    # Libraries for rpy2 and R integration
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libbz2-dev \
    libffi-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    zlib1g-dev \
    # Libraries for graphics and fonts
    libcairo2-dev \
    libxt-dev \
    # System utilities
    wget \
    curl \
    git \
    unzip \
    parallel \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Upgrade pip and install setuptools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install R package manager (renv) globally
RUN Rscript -e "install.packages('renv', repos = c(CRAN = 'https://cloud.r-project.org'))"

# MOSEK setup - Download and install MOSEK 10.2.5 (x86-64 for compatibility)
RUN mkdir -p /opt/mosek && cd /opt/mosek \
    && wget -q https://download.mosek.com/stable/10.2.5/mosektoolslinux64x86.tar.bz2 \
    && tar -xjf mosektoolslinux64x86.tar.bz2 \
    && rm mosektoolslinux64x86.tar.bz2

# Set MOSEK environment variables
ENV MOSEKPATH=/opt/mosek/mosek/10.2/tools
ENV PATH=$MOSEKPATH/bin:$PATH
ENV LD_LIBRARY_PATH=$MOSEKPATH/bin:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Copy dependency files first for better Docker layer caching
COPY requirements.txt .
COPY renv.lock .
COPY .Rprofile .
# Copy essential renv files
COPY renv/activate.R renv/activate.R
COPY renv/settings.json renv/settings.json

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir psutil

# Initialize renv and install R dependencies
RUN Rscript -e "renv::init(bare = TRUE); renv::consent(provided = TRUE); renv::restore()"

# Install Rmosek with correct attach_builder process
RUN RMOSEKDIR="/opt/mosek/mosek/10.2/tools/platform/linux64x86/rmosek" \
    && echo "Using RMOSEKDIR: $RMOSEKDIR" \
    && Rscript -e "source('$RMOSEKDIR/builder.R'); attachbuilder(); install.rmosek()"

# Copy the project files (data/ excluded by .dockerignore)
COPY . .

# Make shell scripts executable
RUN chmod +x *.sh

# Create directories for outputs
RUN mkdir -p data/processed data/simulated_posterior_means results assets logs

# Set environment variables for the application
ENV NUM_CORES=4
ENV PYTHONPATH=/app

# Default command opens an interactive shell
CMD ["/bin/bash"]

# Expose common ports (optional)
EXPOSE 8888

# Add labels for documentation
LABEL maintainer="Empirical Bayes Replication" \
      description="Docker environment for Empirical Bayes replication package" \
      version="1.0" \
      python.version="3.10" \
      r.version="4.4.0" \
      mosek.version="10.2.5"
