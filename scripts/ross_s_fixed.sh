seeds=(2023)

# All mask types
#masks=(dozer dozer_ext_only dozer_ext_0 dozer_ext_null dozer_AND_ext extreme_mask)
masks=(dozer)
#masks=(dozer_ext_only dozer_ext_0 extreme_mask)

lr=5e-5
#lr=0.001
model=dozerformer_Linear
patch_size=60
batch_size=48
# Dozer attention parameters
local_window=2
stride=4
vary_len=1
train=1
fusion=SUM
data=Ross
exp=$data
anorm_thres=(0.7 0.8 0.9 0.75 0.85 0.65 0.66 0.67 0.68 0.69 0.71 0.72 0.73 0.74 0.76 0.77 0.78 0.79 0.8)
patches_thes=(1)

# 96 192 336 720
  for anorm_thre in "${anorm_thres[@]}"
  do
  for patch_thres in "${patches_thes[@]}"
  do
    for mask in "${masks[@]}"
    do
      for seed in "${seeds[@]}"
      do
        echo "================================================================"
        echo "Running Seed: $seed | Mask: $mask | patch_thres | $patch_thres"
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
          --anorm_thres $anorm_thre
        done
      done
    done
  done
done
printf '\a'
