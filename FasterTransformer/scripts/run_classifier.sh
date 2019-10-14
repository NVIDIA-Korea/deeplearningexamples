#!/bin/bash

TASK_NAME=${1:-"MRPC"}
MODE=${2:-"base"} # base or large
PRECISION=${3:-"half"}
BATCH_SIZE=${4:-1}
SEQ_LEN=${5:-128}
HEAD_NUM=${6:-12}
SIZE_PER_HEAD=${7:-64}
OUTPUT_DIR=${8:-"mrpc_output"}

DOCKER_IMAGE=${IMAGE:-"hanjack/bert:cuda10.0-trt6"}
CONTAINER_NAME="bert_trt"

if [ "${PRECISION}" == "half" ]; then
    PRECISION=16
else
    PRECISION=32
fi

CHECKPOINT_PRECISION=""
if [ "${PRECISION}" == "16" ]; then
    CHECKPOINT_PRECISION = "-fp16"
fi

# host data path
DATA_DIR=${DATA_DIR:-"/raid/datasets/bert_tf"}

# container interal paths begin with '/data'
GLUE_DIR=${GLUE_DIR:-"/data/download"}
BERT_BASE_DIR=${BERT_BASE_DIR:-"/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12"}
BERT_LARGE_DIR=${BERT_LARGE_DIR:-"/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16"}

PRETRAINED_DIR=${BERT_BASE_DIR}
if [ mode == "large" ]; then
    PRETRAINED_DIR=${BERT_LARGE_DIR}
fi

# run container
docker_cmd="docker run --rm --name ${CONTAINER_NAME} \
    --net=host --ipc=host --uts=host --ulimit stack=67108864 --ulimit memlock=-1 \
    -v /raid/datasets/bert_tf/:/data \
    ${DOCKER_IMAGE} sleep infinity"

# finding optimal gemm algorithm from the given attention size
init_cmd="docker exec -ti ${CONTAINER_NAME} \
            gemm_fp${PRECISION} ${BATCH_SIZE} ${SEQ_LEN} ${HEAD_NUM} ${SIZE_PER_HEAD}"

# do inference
infer_cmd="docker exec -ti ${CONTAINER_NAME} 
            python run_classifier.py   --task_name=${TASK_NAME}   --do_eval=True  \
                --data_dir=$GLUE_DIR/${TASK_NAME}   \
                --vocab_file=$PRETRAINED_DIR/vocab.txt   \
                --bert_config_file=$PRETRAINED_DIR/bert_config.json   \
                --init_checkpoint=$PRETRAINED_DIR/bert_model.ckpt${CHECKPOINT_PRECISION}   \
                --max_seq_length=${SEQ_LEN}   \
                --eval_batch_size=${BATCH_SIZE}   \
                --output_dir=${OUTPUT_DIR}   \
                --floatx=float${PRECISION} \
                --use_fp16
                --use_xla"

# terminates container
finish_cmd="docker rm -f bert_trt"

echo $docker_cmd
# $docker_cmd &
# sleep 5
echo $init_cmd
# $init_cmd
echo $infer_cmd
# $infer_cmd
echo $finish_cmd
# $finish_cmd

# python ckpt_type_convert.py --init_checkpoint=mrpc_output/model.ckpt-343 --fp16_checkpoint=mrpc_output/fp16_model.ckpt
