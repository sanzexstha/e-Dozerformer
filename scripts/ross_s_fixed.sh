seeds=(2023)

# All mask types
#masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
masks=(dozer_ext_0)
#masks=(dozer_ext_only dozer_ext_0 extreme_mask)
#attns=(prob FedAttn AutoCorr)
attns=(FedAttn AutoCorr)

lr=5e-5
#lr=0.001
model=dozerformer_Linear
patch_size=12
batch_size=24
# Dozer attention parameters
local_window=2
stride=4
vary_len=1
train=1
fusion=SUM
data=Ross
exp=ablation
anorm_thres=(0.9)
patches_thres=(1)

gpu=0
devices=0

# 96 192 336 720
  for anorm_thre in "${anorm_thres[@]}"
  do
  for patch_thres in "${patches_thres[@]}"
  do
    for mask in "${masks[@]}"
    do
      for attn in "${attns[@]}"
      do
        for seed in "${seeds[@]}"
        do
          echo "================================================================"
          echo "Running Seed: $seed | Mask: $mask | "
          echo "================================================================"

          for pred_len in 288
          do
            echo "Prediction Length: $pred_len"

            python run.py \
            --seed $seed \
            --exp_run $exp \
            --batch_size $batch_size \
            --data $data \
            --is_training $train \
            --model $model \
            --moving_avg '13, 17' \
            --seq_len 1440 \
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
            --fusion $fusion \
            --gpu $gpu \
            --devices $devices \
            --anorm_thres $anorm_thre \
            --attn $attn
          done
        done
      done
    done
  done
done
printf '\a'
