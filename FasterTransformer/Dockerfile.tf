FROM nvcr.io/nvidia/tensorflow:19.06-py3

# build tensorflow whl
WORKDIR /opt/tensorflow
RUN bash /opt/tensorflow/nvbuild.sh --python3.5 --noclean
