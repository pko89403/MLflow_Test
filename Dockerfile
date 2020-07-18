FROM continuumio/miniconda3
MAINTAINER seokwooKang <pko954@gmail.com>

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ENV MLFLOW_TRACKING_URI http://host:port


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         git \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install mlflow

COPY ./ /opt/program
WORKDIR /opt/program

ENTRYPOINT ["mlflow", "run", ".", "-e"]
