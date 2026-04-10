#!/bin/bash

# Fixed Parameters
model=dozerformer_Linear
lr=5e-5
batch_size=24
local_window=2
stride=4
vary_len=1
train=1
fusion=SUM
notes=None
dataset=SFC
exp=ablation
gpu=1
devices=1
seq_len=1440
label_len=96
embed_dim=15
moving_avg='13, 17'

# Sweep Parameters (edit these)
combos=$(python3 -c "
from itertools import product
seeds        = [2023]
masks        = ['dozer']
attns        = ['prob', 'FedAttn', 'AutoCorr']
patch_sizes  = [60]
patch_thres  = [1]
pred_lens    = [288]
anorm_thres  = [0.65]
for c in product(seeds, masks, attns, patch_sizes, patch_thres, pred_lens, anorm_thres):
    print('|'.join(map(str, c)))
")

# ── Run Experiments ──
while IFS='|' read -r seed mask attn patch_size patch_thres pred_len anorm_thres; do
  echo "================================================================"
  echo "Seed: $seed | Mask: $mask | Attn: $attn | Pred: $pred_len"
  echo "================================================================"

  python run.py \
    --seed "$seed" \
    --batch_size "$batch_size" \
    --data "$dataset" \
    --is_training "$train" \
    --model "$model" \
    --moving_avg "$moving_avg" \
    --seq_len "$seq_len" \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --embed_dim "$embed_dim" \
    --learning_rate "$lr" \
    --patch_size "$patch_size" \
    --local_window "$local_window" \
    --stride "$stride" \
    --vary_len "$vary_len" \
    --mask "$mask" \
    --patch_thres "$patch_thres" \
    --gpu "$gpu" \
    --devices "$devices" \
    --notes "$notes" \
    --fusion "$fusion" \
    --anorm_thres "$anorm_thres" \
    --exp_run "$exp" \
    --attn "$attn"
done <<< "$combos"

printf '\a'


#seeds=(2023)
#
## All mask types
##masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
#masks=(extreme_mask dozer)
##masks=(dozer_ext_only dozer_ext_0 extreme_mask)
#attns=(prob FedAttn AutoCorr)
#lr=5e-5
##lr=0.001
#model=dozerformer_Linear
#patch_sizes=(60)
#batch_size=24
## Dozer attention parameters
#local_window=2
#stride=4
#vary_len=1
#train=1
#fusion=SUM
#anorm_thres=(0.65)
#notes=None
#dataset=SFC
#exp=ablation
#
#gpu=1
#devices=1
##$(seq 1 $patch_size)  # 1..8 for patch 8, 1..16 for patch 16
#
#for anorm_thres in "${anorm_thres[@]}"
#do
#  for patch_size in "${patch_sizes[@]}"
#  do
#    for patch_thres in 1
#    do
#      for mask in "${masks[@]}"
#      do
#        for attn in "${attns[@]}"
#        do
#        for seed in "${seeds[@]}"
#        do
#          echo "================================================================"
#          echo "Running Seed: $seed | Mask: $mask | patch_thres: $patch_thres / $patch_size | patch_size: $patch_size"
#          echo "================================================================"
#
#          for pred_len in 288
#          do
#            echo "Prediction Length: $pred_len"
#
#            python run.py \
#            --seed $seed \
#            --batch_size $batch_size \
#            --data $dataset \
#            --is_training $train \
#            --model $model \
#            --moving_avg '13, 17' \
#            --seq_len 1440 \
#            --label_len 96 \
#            --pred_len $pred_len \
#            --embed_dim 15 \
#            --learning_rate $lr \
#            --patch_size $patch_size \
#            --local_window $local_window \
#            --stride $stride \
#            --vary_len $vary_len \
#            --mask $mask \
#            --patch_thres $patch_thres \
#            --gpu $gpu \
#            --devices $devices \
#            --notes "$notes" \
#            --fusion $fusion \
#            --anorm_thres $anorm_thres \
#            --exp_run "$exp" \
#            --attn $attn
#          done
#        done
#        done
#      done
#    done
#  done
#done
#printf '\a'