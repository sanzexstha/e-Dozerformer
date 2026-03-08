seeds=(2023)

# All mask types
#masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
masks=(dozer)
#masks=(dozer_ext_only dozer_ext_0 extreme_mask)
batch_size=48
#lr=5e-5
lr=0.001
model=dozerformer_Linear
patch_size=8

# Dozer attention parameters
local_window=3
stride=7
vary_len=1

# 96 192 336 720
patches_thes=(1)
for patch_thres in "${patches_thes[@]}"
do
  for mask in "${masks[@]}"
  do
    for seed in "${seeds[@]}"
    do
      echo "================================================================"
      echo "Running Seed: $seed | Mask: $mask | patch_thres | $patch_thres"
      echo "================================================================"

      for pred_len in 72
      do
        echo "Prediction Length: $pred_len"

        python run.py \
        --seed $seed \
        --batch_size $batch_size \
        --data Lexington \
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
        --patch_thres $patch_thres
      done
    done
  done
done
printf '\a'