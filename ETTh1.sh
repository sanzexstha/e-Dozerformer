## Random Seeds
##seeds=(1 2022 2023 2024 2025 2026)
#seeds=(2023)
#lr=5e-5
#model=dozerformer_Linear
#patch_size=24
## Dozer attention parameters
#local_window=3
#stride=7
#vary_len=1
## shellcheck disable=SC2068
#for seed in ${seeds[@]}
#do
#      echo $seed
#      #----------------------------------predict length 96---------------------------------------
#      python run.py --seed $seed --data ETTh1_labeled --model $model --moving_avg '13, 17' \
#      --seq_len 720 --label_len 96 --pred_len 96 --embed_dim 8 --wandb true \
#      --learning_rate $lr --patch_size $patch_size \
#      --local_window $local_window --stride $stride --vary_len $vary_len
#
#      #----------------------------------predict length 192---------------------------------------
#      python run.py --seed $seed --data ETTh1_labeled --model $model --moving_avg '13, 17' \
#      --seq_len 720 --label_len 96 --pred_len 192 --embed_dim 8 --wandb true \
#      --learning_rate $lr --patch_size $patch_size \
#      --local_window $local_window --stride $stride --vary_len $vary_len
##
##      #----------------------------------predict length 336---------------------------------------
#      python run.py --seed $seed --data ETTh1_labeled --model $model --moving_avg '13, 17' \
#      --seq_len 720 --label_len 96 --pred_len 336 --embed_dim 8 --wandb true \
#      --learning_rate $lr --patch_size $patch_size \
#      --local_window $local_window --stride $stride --vary_len $vary_len
##
##      #----------------------------------predict length 720---------------------------------------
#      python run.py --seed $seed --data ETTh1_labeled --model $model --moving_avg '13, 17' \
#      --seq_len 720 --label_len 96 --pred_len 720 --embed_dim 8 --wandb true \
#      --learning_rate $lr --patch_size $patch_size \
#      --local_window $local_window --stride $stride --vary_len $vary_len
#done
#

# Random Seeds
seeds=(2023)
fusions=(SUM EIA ADT)

lr=5e-5
model=dozerformer_Linear
patch_size=24
local_window=3
stride=7
vary_len=1

for fusion in ${fusions[@]}
do
  echo "Running fusion: $fusion"

  for seed in ${seeds[@]}
  do
    echo "Seed: $seed"

    # -------------------- pred 96 --------------------
    python run.py --seed $seed --fusion $fusion \
      --data ETTh1_labeled --model $model --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 96 --embed_dim 8 --wandb true \
      --learning_rate $lr --patch_size $patch_size \
      --local_window $local_window --stride $stride --vary_len $vary_len

    # -------------------- pred 192 --------------------
    python run.py --seed $seed --fusion $fusion \
      --data ETTh1_labeled --model $model --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 192 --embed_dim 8 --wandb true \
      --learning_rate $lr --patch_size $patch_size \
      --local_window $local_window --stride $stride --vary_len $vary_len

    # -------------------- pred 336 --------------------
    python run.py --seed $seed --fusion $fusion \
      --data ETTh1_labeled --model $model --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 336 --embed_dim 8 --wandb true \
      --learning_rate $lr --patch_size $patch_size \
      --local_window $local_window --stride $stride --vary_len $vary_len

    # -------------------- pred 720 --------------------
    python run.py --seed $seed --fusion $fusion \
      --data ETTh1_labeled --model $model --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 720 --embed_dim 8 --wandb true \
      --learning_rate $lr --patch_size $patch_size \
      --local_window $local_window --stride $stride --vary_len $vary_len

  done
done




