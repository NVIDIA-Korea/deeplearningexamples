#!/bin/bash

CONT=${1:-"hanjack/bert:cuda10.0-trt5"}

docker build . -t ${CONT} -f Dockerfile.cuda10.0-trt5

# docker push ${CONT}