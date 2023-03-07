## Use the container

```Shell
cd docker/

# Build:
docker build --build-arg USER_ID=$UID -t yolact_edge_image .

# Launch (with GPUs):
./start.sh /path/to/yolact_edge /path/to/datasets

# Launch (without GPUs):
./start_cpu.sh /path/to/yolact_edge /path/to/datasets
```
## Run the evaluation

```Shell
python eval_v2.py --config=yolact_resnet50_qualitex_custom_2_config --trained_model=./yolact_edge/weights/yolact_plus_resnet50_qualitex_custom_2_121_115000.pth --images=./eval_dataset/:./results/ --score_threshold=0.2
```

## Run the inference
```Shell
python run_inference.py
```
