#!/usr/bin/env bash
# PASCAL-5i setup for Google Colab. Run from the repo root:
#   bash tools/colab_pascal5i.sh
set -euo pipefail

VOC_TAR="VOCtrainval_11-May-2012.tar"
VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/${VOC_TAR}"

# 1) download + extract VOC2012 (train/val)
if [ ! -d "data/VOC2012/JPEGImages" ]; then
  wget -q --show-progress "${VOC_URL}"
  mkdir -p data
  tar -xf "${VOC_TAR}" -C data
  mv data/VOCdevkit/VOC2012 data/VOC2012
  rm -rf data/VOCdevkit "${VOC_TAR}"
fi

# 2) dependencies (torch/torchvision already present on Colab)
python -m pip install -q -r requirements.txt

# 3) build PASCAL-5i lists + configs from data/splits/pascal
python tools/prepare_pascal5i.py --voc_root data/VOC2012

echo
echo "Done. Train split 0 (1-shot, resnet50):"
echo "python train.py --config config/pascal/pascal_split0_resnet50_MGANet.yaml"
