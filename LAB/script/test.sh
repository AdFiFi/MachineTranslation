#!/bin/bash
cd ../
torchrun  --nproc_per_node=4  main.py \
--datasets "wmt14" \
--log_dir "./log_dir" \
--model "Transformer" \
--task "wmt14_de_en" \
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
--do_parallel \
--train_batch_size 80 \
--num_epochs 10 \
--learning_rate 5e-5 \
--beta1 0.9 \
--beta2 0.98 \
--epsilon 1e-9 \
--schedule 'linear' \
--warmup_steps 4000 \
--save_steps 2000 \
--test_steps 2000 \
--epsilon_ls 0.1 \
\
--do_test \
--evaluate_batch_size 32 \
--num_beams 1 \
--alpha 0.6

torchrun  --nproc_per_node=4  main.py \
--datasets "wmt14" \
--log_dir "./log_dir" \
--model "Stack" \
--task "wmt14_de_en" \
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
--do_parallel \
--train_batch_size 80 \
--num_epochs 10 \
--learning_rate 5e-5 \
--beta1 0.9 \
--beta2 0.98 \
--epsilon 1e-9 \
--schedule 'linear' \
--warmup_steps 4000 \
--save_steps 2000 \
--test_steps 2000 \
--epsilon_ls 0.1 \
\
--do_test \
--evaluate_batch_size 32 \
--num_beams 1 \
--alpha 0.6

### torchrun  --nproc_per_node=4  main.py \
### --datasets "wmt14" \
### --log_dir "./log_dir" \
### --model "Cube" \
### --task "wmt14_de_en" \
### --data_dir "./data" \
### --src_language "de" \
### --tgt_language "en" \
### --data_processors 10 \
### \
### --d_model 512 \
### --num_heads 8 \
### --dim_feedforward 2048 \
### --num_encoder_layers 6 \
### --num_decoder_layers 6 \
### --max_seq_len 128 \
### --activation 'gelu' \
### --model_dir 'output_dir' \
### \
### --do_parallel \
### --train_batch_size 80 \
### --num_epochs 10 \
### --learning_rate 5e-5 \
### --beta1 0.9 \
### --beta2 0.98 \
### --epsilon 1e-9 \
### --schedule 'linear' \
### --warmup_steps 4000 \
### --save_steps 2000 \
### --test_steps 2000 \
### --epsilon_ls 0.1 \
### \
### --do_test \
### --evaluate_batch_size 32 \
### --alpha 0.6