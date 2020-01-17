#!/bin/bash

CONTAINER_NAME=bert_trt

# run container
docker_cmd="docker run --rm -ti --name ${CONTAINER_NAME} 
    --net=host --ipc=host --uts=host --ulimit stack=67108864 --ulimit memlock=-1 
    -v /raid/datasets/bert_tf/:/data 
    hanjack/bert:cuda10.0-trt5 bash"

echo $docker_cmd
$docker_cmd
