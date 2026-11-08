"""Build PASCAL-5i lists + configs for MGANet.

Reads the BAM-style split files under ``data/splits/pascal/{trn,val}/fold*.txt``
(line format: ``<image_id>__<class_id>``, class_id is the VOC mask value 1..20)
and writes the pre-generated lists that ``utils/dataset.py`` expects:

    lists/pascal/fss_list/train/data_list_<split>.txt
    lists/pascal/fss_list/train/sub_class_file_list_<split>.txt
    lists/pascal/fss_list/val/data_list_<split>.txt
    lists/pascal/fss_list/val/sub_class_file_list_<split>.txt
    config/pascal/pascal_split<split>_resnet50_MGANet.yaml

Split convention (standard PASCAL-5i):
    split f -> novel classes = 5*f+1 .. 5*f+5
               train on the other 3 folds (15 base classes)

Usage:
    python tools/prepare_pascal5i.py --voc_root data/VOC2012
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, cls = line.split("__")
            pairs.append((name, int(cls)))
    return pairs


def build(pairs, voc_root):
    """Return (data_list, classwise) with repo-relative POSIX paths."""
    data_list = []
    seen_img = set()
    classwise = {}
    for name, cls in sorted(pairs):
        img = "{}/JPEGImages/{}.jpg".format(voc_root, name).replace("\\", "/")
        mask = "{}/SegmentationClass/{}.png".format(voc_root, name).replace("\\", "/")
        pair = (img, mask)
        if name not in seen_img:
            seen_img.add(name)
            data_list.append(pair)
        lst = classwise.setdefault(cls, [])
        if pair not in lst:
            lst.append(pair)
    return data_list, classwise


def write_list(path, data_list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for img, mask in data_list:
            f.write("{} {}\n".format(img, mask))


def write_classwise(path, classwise):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(repr(classwise))


CONFIG_TEMPLATE = """Data:
  data_root: {voc_root}
  base_data_root: ./data/base_annotation/pascal/
  train_list: ./lists/pascal/fss_list/train/data_list_{split}.txt
  val_list: ./lists/pascal/fss_list/val/data_list_{split}.txt
  classes: 2

Train:
  methods: 'Multi-Similarity-MFG'
  train_h: 473
  train_w: 473
  val_size: 473
  scale_min: 0.9
  scale_max: 1.1
  rotate_min: -10
  rotate_max: 10
  ignore_label: 255
  padding_label: 255
  ppm_scales: [1.0]
  count: 3
  split: {split}
  shot: 1
  aplha: 0.7
  beta: 0.3
  data_set: 'pascal'
  use_split_coco: False
  batch_size: 4
  base_lr: 0.004
  epochs: 200
  start_epoch: 0
  stop_interval: 175
  index_split: -1
  power: 0.9
  momentum: 0.9
  weight_decay: 0.0001
  warmup: False
  print_freq: 10
  save_freq: 50
  resume:
  evaluate: True
  SubEpoch_val: True
  fix_random_seed_val: True
  batch_size_val: 1
  resized_val: True
  ori_resize: True
  workers: 2
  fix_bn: True
  manual_seed: 321
  test_seed: 321
  seed_deterministic: False
  zoom_factor: 8
  amp: False
  amp_dtype: 'bfloat16'

Method:
  layers: 50
  vgg: False
  aux_weight: 1.0
  thresh_fp: 0.7
  thresh_bp: 0.6
  fuse_select: True
  start_loss: 0.8

Test_Finetune:
  weight: ""
  proto_dict_path: "./exp/MGANet_iSAID/pascal/MGANet/split{split}/resnet50/snapshotproto_dict.pth"
  ann_type: 'mask'
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_root", default="data/VOC2012",
                    help="VOC2012 root containing JPEGImages/ and SegmentationClass/")
    ap.add_argument("--splits_dir", default=os.path.join(ROOT, "data", "splits", "pascal"))
    args = ap.parse_args()

    voc_root = args.voc_root.replace("\\", "/")
    if not os.path.isdir(os.path.join(ROOT, args.splits_dir)):
        sys.exit("split dir not found: {}".format(args.splits_dir))

    needed = [
        os.path.join(args.splits_dir, "trn", "fold{}.txt".format(f)) for f in range(4)
    ] + [
        os.path.join(args.splits_dir, "val", "fold{}.txt".format(f)) for f in range(4)
    ]
    missing = [p for p in needed if not os.path.isfile(p)]
    if missing:
        sys.exit("missing split files:\n  " + "\n  ".join(missing))

    for split in range(4):
        trn_pairs = []
        for fold in range(4):
            if fold == split:
                continue
            trn_pairs += read_pairs(os.path.join(args.splits_dir, "trn", "fold{}.txt".format(fold)))
        val_pairs = read_pairs(os.path.join(args.splits_dir, "val", "fold{}.txt".format(split)))

        train_data, train_cw = build(trn_pairs, voc_root)
        val_data, val_cw = build(val_pairs, voc_root)

        novel = set(range(5 * split + 1, 5 * split + 6))
        base = set(range(1, 21)) - novel
        assert set(train_cw) == base, (split, "train classes", sorted(train_cw))
        assert set(val_cw) == novel, (split, "val classes", sorted(val_cw))
        assert all(len(v) >= 2 for v in train_cw.values()), "train class with <2 imgs"
        assert all(len(v) >= 2 for v in val_cw.values()), "val class with <2 imgs"

        tdir = os.path.join(ROOT, "lists", "pascal", "fss_list", "train")
        vdir = os.path.join(ROOT, "lists", "pascal", "fss_list", "val")
        write_list(os.path.join(tdir, "data_list_{}.txt".format(split)), train_data)
        write_classwise(os.path.join(tdir, "sub_class_file_list_{}.txt".format(split)), train_cw)
        write_list(os.path.join(vdir, "data_list_{}.txt".format(split)), val_data)
        write_classwise(os.path.join(vdir, "sub_class_file_list_{}.txt".format(split)), val_cw)

        cfg_dir = os.path.join(ROOT, "config", "pascal")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, "pascal_split{}_resnet50_MGANet.yaml".format(split))
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(CONFIG_TEMPLATE.format(voc_root=voc_root, split=split))

        print("split {}: train imgs={} ({} base classes) | val imgs={} ({} novel classes)"
              .format(split, len(train_data), len(train_cw), len(val_data), len(val_cw)))

    print("done -> lists/pascal/** and config/pascal/**")


if __name__ == "__main__":
    main()
