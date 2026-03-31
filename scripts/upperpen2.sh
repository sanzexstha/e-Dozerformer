seeds=(2023)

# All mask types
#masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
masks=(dozer_ext_0)
#masks=(dozer_ext_only dozer_ext_0 extreme_mask)

lr=5e-5
#lr=0.001
model=dozerformer_Linear
patch_sizes=(60)
batch_size=24
# Dozer attention parameters
local_window=2
stride=4
vary_len=1
train=1
fusion=SUM
anorm_thres=(0.2 0.18 0.16 0.18 0.16 0.14 0.12 0.1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9)
notes=None
dataset=UpperPen
exp=$dataset

gpu=1
devices=1
#$(seq 1 $patch_size)  # 1..8 for patch 8, 1..16 for patch 16

for anorm_thres in "${anorm_thres[@]}"
do
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

          for pred_len in 288
          do
            echo "Prediction Length: $pred_len"

            python run.py \
            --seed $seed \
            --batch_size $batch_size \
            --data $dataset \
            --is_training $train \
            --model $model \
            --moving_avg '13, 17' \
            --seq_len 1440 \
            --label_len 96 \
            --pred_len $pred_len \
            --embed_dim 15 \
            --learning_rate $lr \
            --patch_size $patch_size \
            --local_window $local_window \
            --stride $stride \
            --vary_len $vary_len \
            --mask $mask \
            --patch_thres $patch_thres \
            --gpu $gpu \
            --devices $devices \
            --notes "$notes" \
            --fusion $fusion \
            --anorm_thres $anorm_thres \
            --exp "$exp"
          done
        done
      done
    done
  done
done
printf '\a'