# Random Seeds
seeds=(2023)

# All mask types
#masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
masks=(dozer_ext_local)

lr=1e-4
model=dozerformer_Linear
patch_size=24

# Dozer attention parameters
local_window=3
stride=7
vary_len=1

for mask in "${masks[@]}"
do
  for seed in "${seeds[@]}"
  do
    echo "===================================================="
    echo "Dataset: ETTh2 | Seed: $seed | Mask: $mask"
    echo "===================================================="

    for pred_len in 96 192 336 720
    do
      echo "Prediction Length: $pred_len"

      python run.py \
      --seed $seed \
      --data ETTh2_labeled \
      --model $model \
      --moving_avg '13, 17' \
      --seq_len 720 \
      --label_len 96 \
      --pred_len $pred_len \
      --embed_dim 8 \
      --learning_rate $lr \
      --patch_size $patch_size \
      --local_window $local_window \
      --stride $stride \
      --vary_len $vary_len \
      --mask $mask
    done

  done
done
