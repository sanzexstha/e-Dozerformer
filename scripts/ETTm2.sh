seeds=(2023)
#masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
masks=(dozer_ext_0 dozer_v1)

lr=5e-5
model=dozerformer_Linear
patch_size=48
patches_thes=(1 2 3 4 5 6 7 8 9 10)
local_window=2
stride=4
vary_len=1

for patch_thres in "${patches_thes[@]}"
do
  for mask in "${masks[@]}"
  do
    for seed in "${seeds[@]}"
    do
        echo "==================================================================="
        echo "Dataset: ETTm2 | Seed: $seed | Mask: $mask | thres: $patch_thres"
        echo "==================================================================="
      for pred_len in 96 192 336 720
      do
        python run.py \
        --seed "$seed" \
        --data ETTm2_labeled \
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
