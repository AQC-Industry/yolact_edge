#!/bin/bash

SOURCE_CODE=$1
DATASETS=$2

docker build --target local -t yolact_edge_cpu:latest -f Dockerfile.cpu .

docker run --memory="3g" -it --name=yolact_edge \
  --shm-size=64gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $SOURCE_CODE:/root/yolact_edge/:rw \
  -v $DATASETS:/datasets/:ro \
  yolact_edge_cpu:latest
