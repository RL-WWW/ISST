#!/usr/bin/env bash


python -m experiments.segmentation.test_single_image

# python -m experiments.segmentation.test_single_image \
#     --dataset pcontext \
#     --model deeplab \
#     --jpu JPU \
#     --aux --se-loss \
#     --backbone resnet50 \
#     --resume checkpoints/deeplab_jpu_res50_pcontext.pth.tar \
#     --input-path images/deeplab/input/ \
#     --save-path images/deeplab/output/ \
#     --no-cuda