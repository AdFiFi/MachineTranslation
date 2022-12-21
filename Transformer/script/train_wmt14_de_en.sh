#!/bin/bash
cd ../
python  main.py \
--datasets "wmt14" \
--log_dir "./log_dir" \
--data_dir "./data" \
--src_language "de" \
--tgt_language "en" \
--data_processors 10 \
\
--d_model 512 \
--num_heads 8 \
--dim_feedforward 2048 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--max_seq_len 128 \
--activation 'gelu' \
--model_dir 'output_dir' \
\
--do_train \
--do_parallel \
--train_batch_size 128 \
--num_epochs 3 \
--learning_rate 1e-5 \
--beta1 0.9 \
--beta2 0.98 \
--epsilon 1e-9 \
--schedule 'linear' \
--warmup_steps 4000 \
--save_steps 2000 \
--test_steps 2000 \
--epsilon_ls 0.1 \
\
--do_evaluate \
--evaluate_batch_size 128 \
--alpha 0.6 \
> ./out.log