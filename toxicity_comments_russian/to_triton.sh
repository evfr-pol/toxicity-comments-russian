#!/bin/bash

docker run --gpus all --rm -it \
  -v ./rubert_tiny2_toxic/best_model:/workspace \
  nvcr.io/nvidia/tensorrt:23.08-py3 \
  trtexec --onnx=model.onnx --saveEngine=model.trt
