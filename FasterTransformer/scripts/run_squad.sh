#!/bin/bash

MODE=${1:-"base"} # base or large
BATCH_SIZE=${2:-"1"}
PRECISION=${3:-"fp16"}
SEQ_LEN=${4:-128}
DOC_STRIDE=${5:-"128"}
SQUAD_VERSION=${6:-"1.1"}
HEAD_NUM=${7:-12} # ?
SIZE_PER_HEAD=${8:-64} # ?
OUTPUT_DIR=${9:-"/output_squad"}

DOCKER_IMAGE=${IMAGE:-"hanjack/bert:cuda10.0-trt5"}
CONTAINER_NAME=${CONTAINER_NAME:-"bert_trt"}
GPU_IDX=${GPU_IDX:-0}
DO_PROFILE=${PROFILE:-"0"}
PROFILE_FILENAME=${PROFILE_FILENAME:-"${CONTAINER_NAME}_squad_${MODE}_s${SEQ_LEN}_b${BATCH_SIZE}"}

# host mount path
DATA_DIR=${DATA_DIR:-"/raid/dataset/bert_tf"}

# container internal paths begin with '/data'
SQUAD_DIR=${SQUAD_DIR:-"/data/download/squad/v${SQUAD_VERSION}"}
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
if [ "${PRECISION}" = "fp16" ] ; then
        use_fp16="--use_fp16"
fi

CHECKPOINT_PRECISION=""
if [ "${PRECISION}" == "fp16" ]; then
    CHECKPOINT_PRECISION="-fp16"
fi

# run container
docker_cmd="docker run --rm --name ${CONTAINER_NAME} \
    --net=host --ipc=host --uts=host --ulimit stack=67108864 --ulimit memlock=-1 \
    -e NVIDIA_VISIBLE_DEVICES=${GPU_IDX} \
    -v /raid/datasets/bert_tf/:/data \
    -v /raid/outputs:/${OUTPUT_DIR} \
    ${DOCKER_IMAGE} sleep infinity"

# finding optimal gemm algorithm from the given attention size
init_cmd="docker exec -ti \
        ${CONTAINER_NAME} gemm_${PRECISION} ${BATCH_SIZE} ${SEQ_LEN} ${HEAD_NUM} ${SIZE_PER_HEAD}"

profile_cmd=""
if [ ${DO_PROFILE} == 1 ]; then
    profile_cmd="nsys profile -f true -t cuda,cudnn,cublas,nvtx -o ${PROFILE_FILENAME} "
fi

# do inference
infer_cmd="docker exec -ti ${CONTAINER_NAME} \
        ${profile_cmd} \
        python run_squad.py  \
            --vocab_file=$PRETRAINED_DIR/vocab.txt   \
            --bert_config_file=$PRETRAINED_DIR/bert_config.json   \
            --init_checkpoint=$PRETRAINED_DIR/bert_model.ckpt${CHECKPOINT_PRECISION}   \
            --do_predict = True \
            --predict_file=${SQUAD_DIR}/dev-v${SQUAD_VERSION}.json   \
            --max_seq_length=${SEQ_LEN}   \
            --doc_stride=${DOC_STRIDE} \
            --predict_batch_size=${BATCH_SIZE} \
            --output_dir=${OUTPUT_DIR}   \
            --version_2_with_negative=${version_2_with_negative}
            --num_eval_iterations=1000 \
            ${use_fp16} --use_xla"

eval_cmd="docker exec -ti \
        ${CONTAINER_NAME} \
        python $SQUAD_DIR/evaluate-v${SQUAD_VERSION}.py ${SQUAD_DIR}/dev-v${SQUAD_VERSION}.json ${OUTPUT_DIR}/predictions.json"

# terminates container
finish_cmd="docker rm -f ${CONTAINER_NAME}"

echo $docker_cmd
$docker_cmd &
sleep 2
echo $init_cmd
$init_cmd
echo $infer_cmd
$infer_cmd
if [ ${DO_PROFILE} == 1 ]; then
    docker exec -ti ${CONTAINER_NAME} \
        mv "${PROFILE_FILENAME}.qdrep" "${OUTPUT_DIR}/${PROFILE_FILENAME}.qdrep"
fi

# echo $eval_cmd
# $eval_cmd
echo $finish_cmd
$finish_cmd

