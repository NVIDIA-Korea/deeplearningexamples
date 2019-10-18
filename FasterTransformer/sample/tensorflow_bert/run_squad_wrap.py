# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# usage example
# export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
# export SQUAD_DIR=/path/to/glue
# python run_squad_wrap.py   --use_fp16   --task_name=MRPC   --do_eval=true   --data_dir=$GLUE_DIR/MRPC   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=mrpc_output/fp16_model.ckpt   --max_seq_length=128   --eval_batch_size=8   --output_dir=squad_output

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf

# bert_submodule = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert')
bert_submodule = "/opt/FasterTransformer/sample/tensorflow_bert"
sys.path.insert(0, bert_submodule)
import my_modeling
import fast_infer_util as fiu
import run_squad

flags = tf.flags
FLAGS = flags.FLAGS

# replace transformer implementation
my_modeling.transformer_model = fiu.transformer_model
# replace the model to support fp16 data type
run_squad.create_model = fiu.create_squad_model
# replace the input function to drop remainderfile_based_input_fn_builder = fiu.file_based_input_fn_builder_drop
main = run_squad.main

if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
