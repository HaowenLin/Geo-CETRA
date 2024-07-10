# coding=utf-8
#!/bin/bash


python app.py \
    --is_wandb_used=False \
    --time_decoder=none \
    --best_model_cp=/home/users/constraint_gen/exps/thp/baseline_60mins/11_22_2023-16_05_22 \
    --training_mode=training \
    --test_small_batch=False \
    --dataset=baseline_60mins_constraint_polar \
    --save_path=/home/users/constraint_gen/exps \
    --display_name=time_only_pretime_base \
    --lr=0.01 \
    --is_cuda=1 \
    --cuda_id=0 \
    --batch_size=256 \
    --normalization=none \
    --epochs=250 \
    --time_format=time_gap \
    --mode=none \
    --gen_epoch=1000 \
    --model_choice=time_log \
    --spatial_num_mix_components=8 \
    --num_mix_components=1 \
    --scheduler_gamma=0.99 \
    --n_head=8 \
    --lr_s_std_regularization=0.001 \
    --spatial_model=condGMM