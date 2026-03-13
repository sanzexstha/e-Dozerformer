#masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
masks=(dozer_ext_0)

# Random Seeds
# ~15 minutes
seeds=(2023)

lr=5e-5
model=dozerformer_Linear
patch_size=24
#patches_thes=(1 2 3 4 5 6 7 8 9 10)
patches_thes=(24)
batch_size=32
gpu=0
devices=0
#96 192 336 720
# Dozer attention parameters
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
      echo "Dataset: ETTm1 | Seed: $seed | Mask: $mask | thres: $patch_thres"
      echo "==================================================================="
      for pred_len in 96 192 720
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
        --mask "$mask" \
        --patch_thres "$patch_thres" \
        --gpu "$gpu" \
        --devices "$devices" \
        --batch_size "$batch_size"
      done
    done
  done
done
# After all the loops finish
python -c "import wandb; wandb.init(project='e-Dozerformer'); wandb.alert(title='Training Complete', text='All runs finished!')"