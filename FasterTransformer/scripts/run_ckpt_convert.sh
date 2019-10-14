#!/bin/bash

INPUT_CKPT=${1:-"/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt"}
OUTPUT_CKPT=${2:-"/data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt-fp16"}

python ckpt_type_convert.py --init_checkpoint=${INPUT_CKPT} --fp16_checkpoint=${OUTPUT_CKPT}