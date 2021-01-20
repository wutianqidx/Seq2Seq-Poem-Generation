#!/usr/bin/env bash
python decode.py \
    --gpu_ids "2" \
    --decode_input  data/data.src.test\
    --decode_output data/data.trg.test\
    --model_path "model_2019xxxxxx/seq2seq.ckpt-xxxx" \
    --decode_batch_size 1 \
    --max_decode_step 15 \
    --decode_ignore_unk True \
    --beam_width 5 \


