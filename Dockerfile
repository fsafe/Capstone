FROM ubuntu:18.04

RUN sudo apt update && \
    sudo apt install -yq python3-pip && \
    sudo apt install -yq libsm6 libxext6 libxrender-dev

WORKDIR /deeplesion

COPY requirements.txt ./

RUN PATH=/home/ubuntu/.local/bin:$PATH && \
    export PATH && \
    pip3 install --user --upgrade --no-cache-dir pip -r requirements.txt && \
    pip3 install --user --no-cache-dir opencv-python

COPY . /deeplesion
