#!/usr/bin/env bash
set -e

mkdir -p checkpoints

wget -c \
  -O checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
