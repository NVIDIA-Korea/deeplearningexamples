#!/bin/bash

INPUT_CKPT=${1:-"/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt"}
OUTPUT_CKPT=${2:-"/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt-fp16"}

DOCKER_IMAGE=${IMAGE:-"hanjack/bert:cuda10.0-trt6"}
DATA_DIR=${DATA_DIR:-"/raid/datasets/bert_tf"}

docker run --rm --name bert_trt --net=host --ipc=host --uts=host --ulimit stack=67108864 --ulimit memlock=-1 \
    -v ${DATA_DIR}:/data ${DOCKER_IMAGE} sleep infinity &

docker exec -ti bert_trt \
    python ckpt_type_convert.py --init_checkpoint=${INPUT_CKPT} --fp16_checkpoint=${OUTPUT_CKPT}

docker rm -f bert_trt
