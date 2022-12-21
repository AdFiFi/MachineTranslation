export PYTHONUNBUFFERED=1

cd ../
python  main.py \
--datasets "Multi30k" \
--log_dir "./log_dir" \
--datasets "Multi30k" \
--data_dir "./data" \
--src_language "de" \
--tgt_language "en" \
--data_processors 10 \
\
--d_model 512 \
--num_heads 8 \
--dim_feedforward 1024 \
--num_encoder_layers 3 \
--num_decoder_layers 3 \
--max_seq_len 128 \
--activation 'gelu' \
--model_dir 'output_dir' \
\
--do_train \
--do_parallel \
--device "cuda" \
--train_batch_size 128 \
--num_epochs 5 \
--learning_rate 5e-5 \
--beta1 0.9 \
--beta2 0.98 \
--epsilon 1e-9 \
--schedule 'linear' \
--warmup_steps 1000 \
--save_steps 1000 \
--test_steps 2000 \
--epsilon_ls 0.1 \
\
--do_evaluate \
--evaluate_batch_size 128 \
--beam_num 5 \
--alpha 0.6