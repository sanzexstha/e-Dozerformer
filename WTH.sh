# ~10 mins
# Random Seeds
seeds=(2023)

# Mask types
#masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
masks=(dozer_v1)
#masks=(dozer)

patches_thes=(2 3 4 5)
#patches_thes=(2)

lr=1e-4
model=dozerformer_Linear
patch_size=24

# Dozer attention parameters
local_window=3
stride=3
vary_len=1

for patch_thres in "${patches_thes[@]}"
do
  for mask in "${masks[@]}"
  do
    for seed in "${seeds[@]}"
    do
      echo "===================================================================="
      echo "Dataset: Weather | Seed: $seed | Mask: $mask | thres: $patch_thres"
      echo "===================================================================="

      for pred_len in 96 192 336 720
      do
        echo "Prediction Length: $pred_len"

        python run.py \
        --seed "$seed" \
        --data Weather_labeled \
        --model "$model" \
        --moving_avg '13, 17' \
        --seq_len 720 \
        --label_len 96 \
        --pred_len "$pred_len" \
        --embed_dim 8 \
        --learning_rate "$lr" \
        --patch_size "$patch_size" \
        --local_window "$local_window" \
        --stride "$stride" \
        --vary_len "$vary_len" \
        --mask "$mask" \
        --patch_thres "$patch_thres"
      done
    done
  done
done