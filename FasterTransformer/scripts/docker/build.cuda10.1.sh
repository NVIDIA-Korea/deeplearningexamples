#!/bin/bash
# WARNING: Current Faster Transformer version shows error on CUDA 10.1

CONT=${1:-"nvcr.io/nvidian/sae/jahan:bert_cuda10.1-trt5"}

docker pull nvcr.io/nvidia/tensorflow:19.06-py3

docker build . -f Dockerfile.tf -t tensorflow:1.13

docker build . -t ${CONT} -f Dockerfile

# docker push ${CONT}