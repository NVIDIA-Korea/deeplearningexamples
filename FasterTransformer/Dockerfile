FROM tensorflow:1.13 as tf

FROM nvcr.io/nvidia/tensorrt:19.09-py3 

# install tensorflow
COPY --from=tf /tmp/pip/tensorflow_gpu-*.whl /tmp
RUN pip install /tmp/tensorflow_gpu-*.whl && rm /tmp/tensorflow_gpu-*.whl

# install FasterTransformer
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64/
WORKDIR /opt/FasterTransformer
COPY . .
RUN mkdir -p build && \
    cd build && \
    cmake -DSM=70 \
        -DCMAKE_BUILD_TYPE=Debug \
        -DBUILD_TRT=ON -DTRT_PATH=/usr/lib/x86_64-linux-gnu/ \
        -DBUILD_TF=ON  -DTF_PATH=/usr/local/lib/python3.5/dist-packages/tensorflow .. && \
    make -j && \
    make install
RUN mkdir /workspace/FasterTransformer && \
    mv /opt/FasterTransformer/sample/tensorflow_bert/run_classifier_wrap.py /workspace/FasterTransformer/run_classifier.py && \
    mv /opt/FasterTransformer/sample/tensorflow_bert/run_squad_wrap.py      /workspace/FasterTransformer/run_squad.py && \
    mv /opt/FasterTransformer/sample/tensorflow_bert/profile_transformer_inference.py   /workspace/FasterTransformer/profile_transformer_inference.py && \
    mv /opt/FasterTransformer/sample/tensorflow_bert/ckpt_type_convert.py   /workspace/FasterTransformer/ckpt_type_convert.py

WORKDIR /workspace/FasterTransformer