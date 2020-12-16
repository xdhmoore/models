# TODO parameterize gpu/cpu
#FROM tensorflow/tensorflow:1.15.2-gpu-py3
FROM tensorflow/tensorflow:1.15.2-py3

ARG DEBIAN_FRONTEND=noninteractive

# TODO lock in the version on all this stuff

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

# TODO parameterize
# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
#RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    #echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    #curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    #apt-get update -y && apt-get install google-cloud-sdk -y

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
#WORKDIR /home/tensorflow

# Copy this version of of the model garden into the image
#COPY --chown=tensorflow . /home/tensorflow/models
#RUN rm -rf /home/tensorflow/.envrc
#RUN rm -rf /home/tensorflow/.direnv

# Compile protobuf configs
#RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

# TODO does this really need to be copied to work?
#RUN cp object_detection/packages/tf1/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"
RUN python -m pip install --user -U pip
# TODO add this line back:
#RUN python -m pip install --user .

# Install pip dependencies
#RUN pip3 install --user absl-py
#RUN pip3 install --user contextlib2
#RUN pip3 install --user Cython
#RUN pip3 install --user jupyter
#RUN pip3 install --user matplotlib
#RUN pip3 install --user pycocotools
#RUN pip3 install --user tf-slim

# TODO should these two steps be done outside and mounted instead?
# TODO this is currently copying all gzipped and unzipped data over
#COPY --chown=tensorflow . /home/tensorflow/models/research

# TODO add this step to the instructions for setup outside of docker
#RUN (cd /home/tensorflow/models/research/ && \
#   protoc object_detection/protos/*.proto --python_out=.)


# RESUME: determine how to share /code directory but still compile protoc
# Options:
# 1) Keep as is
# 2) put protoc before dockerfile runs
# 3) put protoc when app starts

# Reuse the layer from above but copy this onto it...?
#COPY --chown=tensorflow object_detection/dockerfiles/tf1/pipeline.config /home/tensorflow/pipeline.config

# TODO put these in a folder
#COPY --chown=tensorflow object_detection/dockerfiles/tf1/run /home/tensorflow/run
#COPY --chown=tensorflow object_detection/dockerfiles/tf1/run-* /home/tensorflow/

#RUN chmod u+x /home/tensorflow/run
#RUN chmod u+x /home/tensorflow/run-*

ENV TF_CPP_MIN_LOG_LEVEL 3

# RESUME set up docker-compose to start up and then you can connect to running image
# RESUME try using non-gpu docker image

# TODO expose tensorboard port
# EXPOSE
# TODO what's the point of this volume declaration?
VOLUME ["/data", "/model_dir" ]

USER root
# TODO clean up this and entrypoint paths
#RUN ln -s /home/tensorflow/models/research/object_detection/dockerfiles/tf1/run /usr/local/bin/run
# TODO this wont work before volume is bound
RUN ln -s /home/tensorflow/models/research/object_detection/dockerfiles/tf1/run /home/tensorflow/.local/bin/run

USER tensorflow

ENTRYPOINT [ "/home/tensorflow/models/research/object_detection/dockerfiles/tf1/run" ]
CMD ["wait"]
#ENTRYPOINT [ "python3", "object_detection/model_main.py",  "--pipeline_config_path=pipeline.config",  "--model_dir=model_dir",  "--num_train_steps=1000",  "--sample_1_of_n_eval_examples=1",  "--alsologtostderr" ]


# TODO commands to make easy:
# build
# start all services (tensorboard?)
# run pipeline
# tweak pipeline and run again
# connect to box
#

# TODO add to instructions:
# chmod ugo+w /host/model_dir
