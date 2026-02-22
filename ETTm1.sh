#masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
masks=(dozer_ext_local)

# Random Seeds
# ~15 minutes
seeds=(2023)

lr=5e-5
model=dozerformer_Linear
patch_size=48

# Dozer attention parameters
local_window=2
stride=4
vary_len=1
for mask in "${masks[@]}"
do
  for seed in "${seeds[@]}"
  do
    for pred_len in 96 192 336 720
    do
      python run.py \
      --seed "$seed" \
      --data ETTm1_labeled \
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
      --mask "$mask"
    done
  done
done
