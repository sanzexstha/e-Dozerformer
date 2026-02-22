# Random Seeds
seeds=(2023)

# Mask types
#masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
masks=(dozer_ext_local)

lr=5e-5
model=dozerformer_Linear
stride=0

for mask in "${masks[@]}"
do
  for seed in "${seeds[@]}"
  do
    echo "================================================"
    echo "Dataset: Exchange | Seed: $seed | Mask: $mask"
    echo "================================================"

    # ---------------- 96 ----------------
    python run.py \
    --seed "$seed" \
    --data Exchange_labeled \
    --model "$model" \
    --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 96 \
    --embed_dim 21 --dropout 0.2 \
    --learning_rate "$lr" \
    --patch_size 48 \
    --loss L1 \
    --local_window 3 \
    --stride "$stride" \
    --vary_len 1 \
    --mask "$mask"

    # ---------------- 192 ----------------
    python run.py \
    --seed "$seed" \
    --data Exchange_labeled \
    --model "$model" \
    --moving_avg '13, 17' \
    --seq_len 96 --label_len 48 --pred_len 192 \
    --embed_dim 21 --dropout 0.2 \
    --learning_rate "$lr" \
    --patch_size 24 \
    --loss L1 \
    --local_window 3 \
    --stride "$stride" \
    --vary_len 1 \
    --mask "$mask"

    # ---------------- 336 ----------------
    python run.py \
    --seed "$seed" \
    --data Exchange_labeled \
    --model "$model" \
    --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 336 \
    --embed_dim 21 --dropout 0.2 \
    --learning_rate "$lr" \
    --patch_size 48 \
    --loss L1 \
    --local_window 3 \
    --stride "$stride" \
    --vary_len 1 \
    --mask "$mask"

    # ---------------- 720 ----------------
    python run.py \
    --seed "$seed" \
    --data Exchange_labeled \
    --model "$model" \
    --moving_avg '13, 17' \
    --seq_len 336 --label_len 96 --pred_len 720 \
    --embed_dim 21 --dropout 0.2 \
    --learning_rate "$lr" \
    --patch_size 48 \
    --loss L1 \
    --local_window 3 \
    --stride "$stride" \
    --vary_len 1 \
    --mask "$mask"

  done
done