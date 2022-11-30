FROM nvcr.io/nvidia/pytorch:22.09-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get -y update
RUN apt-get install -y tzdata
RUN apt install -y tmux
RUN apt install -y htop
RUN apt install -y ncdu

COPY requirements.txt ./
RUN pip install -r requirements.txt && rm ./requirements.txt

WORKDIR /faster-rcnn-inference
