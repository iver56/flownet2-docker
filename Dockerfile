# Note: Our Caffe version does not work with CuDNN 6
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

# Put everything in some subfolder
WORKDIR "/flownet2"

# Install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        module-init-tools \
        build-essential \
        ca-certificates \
        wget \
        git \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-scipy \
        python-pip \
        python-protobuf \
        python-pillow \
        python-skimage && \
    rm -rf /var/lib/apt/lists/*

RUN pip install setuptools==40.4.3 wheel==0.32.1
RUN pip install flask==0.12.2

# Download flownet2 source code
RUN git clone https://github.com/lmb-freiburg/flownet2 && \
    rm -rf flownet2/.git

# Download models. Hundreds of megabytes.
RUN cd flownet2/models && \
    bash download-models.sh && \
    rm flownet2-*.tar.gz && \
    cd .. && cd ..

# The build context contains some files which make the raw FlowNet2
# repo fit for Docker
COPY FN2_Makefile.config ./
COPY FN2_run-flownet-docker.py ./

# Build DispNet/FlowNet Caffe distro
RUN mv ./FN2_Makefile.config ./flownet2/Makefile.config && \
    mv ./FN2_run-flownet-docker.py ./flownet2/scripts/run-flownet-docker.py && \
    cd flownet2 && \
    make -j`nproc` && \
    make -j`nproc` pycaffe && \
    cd ..

RUN pip install uwsgi==2.0.16

# Copy in code for custom web service
COPY web_service.py ./flownet2/scripts/
COPY image_utils.py ./flownet2/scripts/
COPY run.sh ./flownet2/scripts/

WORKDIR "/flownet2/flownet2/scripts"
EXPOSE 5003
CMD ["/bin/bash", "/flownet2/flownet2/scripts/run.sh"]
