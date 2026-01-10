#!/usr/bin/env bash
set -euo pipefail

/home/eshuranov/miniconda3/bin/conda run -n cbramod --no-capture-output \
  python /media/public/eshuranov/cbramod/finetune_main.py \
  --cuda 1 \
  --epochs=50 \
  --batch_size=64 \
  --lr=0.0001 \
  --weight_decay=0.05 \
  --optimizer=AdamW \
  --clip_value=1 \
  --dropout=0.1 \
  --classifier=all_patch_reps \
  --downstream_dataset=CHB-MIT \
  --datasets_dir=/media/public/Datasets/cbramod_data/CHBMIT/processed_seg \
  --num_of_classes=2 \
  --model_dir=/media/public/ckpts/CBR_chkpnts_for_shufle_track/CHBMIT_sh2 \
  --num_workers=16 \
  --label_smoothing=0.1 \
  --multi_lr=True \
  --frozen=False \
  --use_pretrained_weights=True \
  --is_chanle_shafle True \
  --new_order "[13, 15, 1, 8, 10, 0, 2, 3, 11, 6, 9, 14, 7, 4, 5, 12]"
