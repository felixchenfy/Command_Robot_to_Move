#!/bin/bash

data_source="webcam"
image_data_path=""

data_source="video"
image_data_path="data/digits_eval3/video.avi"

data_source="folder"
image_data_path="data/digits_eval2/"

python src/detect_v2.py \
    --weights_path "weights/yolov3_ckpt_20.pth" \
    --model_def    "data/digits_generated/yolo.cfg"\
    --class_path   "data/digits_generated/classes.names" \
    --conf_thres 0.95 \
    --nms_thres 0.1 \
    --batch_size 1 \
    --n_cpu 8 \
    --data_source $data_source \
    --image_data_path $image_data_path
