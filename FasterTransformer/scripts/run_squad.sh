#!/bin/bash

MODE=${1:-"base"} # base or large
BATCH_SIZE=${2:-"1"}
PRECISION=${3:-"half"}
USE_XLA=${4:-"true"}
SEQ_LEN=${5:-128}
DOC_STRIDE=${6:-"128"}
SQUAD_VERSION=${7:-"1.1"}
OUTPUT_DIR=${8:-"output_squad"}
HEAD_NUM=${8:-12} # ?
SIZE_PER_HEAD=${9:-64} # ?

DOCKER_IMAGE=${IMAGE:-"hanjack/bert:cuda10.0-trt6"}
CONTAINER_NAME=bert_trt

# host mount path
DATA_DIR=${DATA_DIR:-"/raid/dataset/bert_tf"}

# container internal paths begin with '/data'
SQUAD_DIR=${SQUAD_DIR:-"/data/download/squad/v${SQUAD_VERSION}/dev-v${SQUAD_VERSION}.json}
BERT_BASE_DIR=${BERT_BASE_DIR:-"/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12"}
BERT_LARGE_DIR=${BERT_LARGE_DIR:-"/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16"}

PRETRAINED_DIR=${BERT_BASE_DIR}
if [ mode == "large" ]; then
    PRETRAINED_DIR=${BERT_LARGE_DIR}
fi

if [ "$SQUAD_VERSION" = "1.1" ] ; then
    version_2_with_negative="False"
else
    version_2_with_negative="True"
fi

use_fp16=""
if [ PRECISION == "half" ] ; then
    echo "fp16 activated!"
    use_fp16="--use_fp16"
else
    PRECISION=32
fi

CHECKPOINT_PRECISION=""
if { "${PRECISION}" == "half" }; then
    CHECKPOINT_PRECISION = "-fp16"
fi

use_xla = ""
if [ "$use_xla" = "true" ] ; then
    echo "XLA activated"
    use_xla = "--use_xla"
fi

# run container
docker_cmd="docker run --rm --name ${CONTAINER_NAME} \
    --net=host --ipc=host --uts=host --ulimit stack=67108864 --ulimit memlock=-1 \
    -v /raid/datasets/bert_tf/:/data \
    ${DOCKER_IMAGE} sleep infinity"

# finding optimal gemm algorithm from the given attention size
init_cmd="docker exec -ti \
        ${CONTAINER_NAME} gemm_fp${PRECISION} ${BATCH_SIZE} ${SEQ_LEN} ${HEAD_NUM} ${SIZE_PER_HEAD}"

# do inference
infer_cmd="docker exec -ti \
        ${CONTAINER_NAME} \
        python run_squad.py  \
            --vocab_file=$PRETRAINED_DIR/vocab.txt   \
            --bert_config_file=$PRETRAINED_DIR/bert_config.json   \
            --init_checkpoint=$PRETRAINED_DIR/bert_model.ckpt${CHECKPOINT_PRECISION}   \
            --do_predict = True \
            --predict_file=$SQUAD_DIR   \
            --max_seq_length=${SEQ_LEN}   \
            --doc_stride=${DOC_STRIDE} \
            --predict_batch_size=${BATCH_SIZE} \
            --output_dir=${OUTPUT_DIR}   \
            --floatx=float${PRECISION} \
            ${use_fp16} ${use_xla} \
            --version_2_with_negative=${version_2_with_negative}"

eval_cmd="docker exec -ti \
        ${CONTAINER_NAME} \
        python $SQUAD_DIR/evaluate-v${SQUAD_VERSION}.py ${SQUAD_DIR}/dev-v${SQUAD_VERSION}.json ${RESULTS_DIR}/predictions.json"

# terminates container
finish_cmd="docker rm -f bert_trt"

echo $docker_cmd
# $docker_cmd &
# sleep 5
echo $init_cmd
# $init_cmd
echo $infer_cmd
# $infer_cmd
echo $eval_cmd
# $eval_cmd
echo $finish_cmd
# $finish_cmd

