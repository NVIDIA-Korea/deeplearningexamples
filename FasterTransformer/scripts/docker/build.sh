#!/bin/bash

CONT=${1:-"nvcr.io/nvidian/sae/jahan:bert-cuda10.0-trt6"}

docker build . -t ${CONT} -f Dockerfile.trt6_cuda10

# docker push ${CONT}