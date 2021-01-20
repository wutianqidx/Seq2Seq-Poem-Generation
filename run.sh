#!/usr/bin/env bash
# Assign gpu ids for parallel training

#encoder_type: uni, bi, emb

python -u train.py \
    --gpu_ids "0,1,2,3" \
    --source_vocabulary data/data.src.dict \
    --target_vocabulary data/data.trg.dict \
    --source_train_data data/data.src.train \
    --target_train_data data/data.trg.train \
    --source_valid_data data/data.src.valid \
    --target_valid_data data/data.trg.valid \
    --dropout_rate 0.2 \
    --batch_size 128 \
    --save_freq 500 \
    --valid_freq 500 \
    --encoder_type emb \
    --hidden_units 128 \
    --depth 2 \
    --cell_type gru \
    --attn_input_feeding False \
 
