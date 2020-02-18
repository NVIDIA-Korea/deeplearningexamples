# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#ARG FROM_IMAGE_NAME=nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

ARG myuid
ARG mygid
RUN echo $myuid $mygid
# TODO: Depending on the docker version, this might work without mapping the user for home dir access
RUN groupadd -r --gid ${mygid} bert && useradd -r -u ${myuid} --gid ${mygid} -ms /bin/bash bert
RUN echo 'bert:bert' | chpasswd
RUN mkdir -p /workspace && chown -R bert /workspace

# Install required libraries
ARG TRT_PKG_VERSION=7.0.0-1+cuda10.2
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    pbzip2 pv bzip2  sudo gcc-7 g++-7  zlib1g-dev  \
    unzip \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libnvinfer6=${TRT_PKG_VERSION} \
    libnvinfer-dev=${TRT_PKG_VERSION} \
    libnvinfer-plugin6=${TRT_PKG_VERSION} \
    libnvinfer-plugin-dev=${TRT_PKG_VERSION} \
    libnvparsers6=${TRT_PKG_VERSION} \
    libnvparsers-dev=${TRT_PKG_VERSION} \
    libnvonnxparsers6=${TRT_PKG_VERSION} \
    libnvonnxparsers-dev=${TRT_PKG_VERSION} \
    python3-libnvinfer=${TRT_PKG_VERSION} \
    python3-libnvinfer-dev=${TRT_PKG_VERSION}

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                                 --slave /usr/bin/g++ g++ /usr/bin/g++-7  && \
                                 update-alternatives --config gcc

# Install Cmake
RUN cd /tmp &&\
   wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh &&\
   chmod +x cmake-3.14.4-Linux-x86_64.sh &&\
   ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license &&\
   rm ./cmake-3.14.4-Linux-x86_64.sh

# Install Tensorflow and pycuda
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN pip install tensorflow==1.14.0 && \
    pip install pycuda

# User-space configuration
WORKDIR /workspace
USER bert

# Download TensorRT OSS
RUN git clone -b release/6.0 https://github.com/nvidia/TensorRT TensorRT && \
   cd TensorRT && \
   git submodule update --init --recursive && \
   mkdir -p demo/BERT/build && cd demo/BERT/build && \
   cmake .. && \
   make -j$(nproc) && \
   tar -czf bert_plugin.tar.gz libbert_plugins.so libcommon.so

# Return to project directory and open a terminal
WORKDIR /workspace

CMD /bin/bash
