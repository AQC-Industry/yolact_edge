# YolactEdge: Real-time Instance Segmentation on the Edge

YolactEdge is an instance segmentation model developed to detect objects in realtime on edge computing systems. In this repository the model was adapted to run an inference module only using CPU for offline applications.

# Model weights
Create a directory called ```./yolact_edge/weights/``` to save the trained weights of the model.

Ask AQC admins for training weights using Qualitex datasets ```yolact_plus_resnet50_qualitex_custom_2_121_115000.pth```

# Dataset
Create a directory inside the repository with the images to do the predictions.

# Development

## How to install model locally

1. Clone project from github
2. Build docker image using target local
```
docker-compose build
```
3. Run docker image specifing the following volumes:
  - Source Code
```
docker run --memory="3g" -it --name=aqc-service-app --shm-size=64gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v SOURCE_CODE:/root/yolact_edge/:rw aqc-service-app:latest
```

## How to install model production
1. Clone project from github
2. Build docker image using target dev
```
docker-compose build
```
3. Run docker image
```
docker run --memory="3g" -it --name=aqc-service-app aqc-service-app:latest
```
4. Stop docker container
```
docker rm aqc-service-app
```
