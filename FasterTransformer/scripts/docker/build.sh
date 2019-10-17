#!/bin/bash

CONT=${1:-"nvcr.io/nvidian/sae/jahan:bert-cuda10.0-trt5"}

docker build . -t ${CONT} -f Dockerfile.trt5_cuda10

docker tag ${CONT} hanjack/bert:cuda10.0-trt5

# docker push ${CONT}