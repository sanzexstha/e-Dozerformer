seeds=(2023)

masks=(dozer)
batch_size=48
lr=0.001
model=dozerformer_Linear
patch_sizes=(8)

# Dozer attention parameters
local_window=3
stride=7
vary_len=1
train=1
gpu=1
devices=1
#$(seq 1 $patch_size)  # 1..8 for patch 8, 1..16 for patch 16
for patch_size in "${patch_sizes[@]}"
do
  for patch_thres in 1
  do
    for mask in "${masks[@]}"
    do
      for seed in "${seeds[@]}"
      do
        echo "================================================================"
        echo "Running Seed: $seed | Mask: $mask | patch_thres: $patch_thres / $patch_size | patch_size: $patch_size"
        echo "================================================================"

        for pred_len in 72
        do
          echo "Prediction Length: $pred_len"

          python run.py \
          --seed $seed \
          --batch_size $batch_size \
          --data Lexington \
          --is_training $train \
          --model $model \
          --moving_avg '13, 17' \
          --seq_len 360 \
          --label_len 96 \
          --pred_len $pred_len \
          --embed_dim 8 \
          --learning_rate $lr \
          --patch_size $patch_size \
          --local_window $local_window \
          --stride $stride \
          --vary_len $vary_len \
          --mask $mask \
          --patch_thres $patch_thres \
          --gpu $gpu \
          --devices $devices
        done
      done
    done
  done
done
printf '\a'