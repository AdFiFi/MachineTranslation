export PYTHONUNBUFFERED=1

cd ../
torchrun  --nproc_per_node=4   main.py \
--datasets "Multi30k" \
--log_dir "./log_dir" \
--model "Stack" \
--task "multi30k_de_en" \
--datasets "Multi30k" \
--data_dir "./data" \
--src_language "de" \
--tgt_language "en" \
--data_processors 10 \
\
--d_model 512 \
--num_heads 8 \
--dim_feedforward 2048 \
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
--num_epochs 20 \
--learning_rate 1e-4 \
--beta1 0.9 \
--beta2 0.98 \
--epsilon 1e-9 \
--schedule 'linear' \
--warmup_steps 100 \
--save_steps 100 \
--test_steps 100 \
--epsilon_ls 0.1 \
\
--do_evaluate \
--do_test \
--evaluate_batch_size 32 \
--num_beams 5 \
--alpha 0.6

torchrun  --nproc_per_node=4   main.py \
--datasets "Multi30k" \
--log_dir "./log_dir" \
--model "Cube" \
--task "multi30k_de_en" \
--datasets "Multi30k" \
--data_dir "./data" \
--src_language "de" \
--tgt_language "en" \
--data_processors 10 \
\
--d_model 512 \
--num_heads 8 \
--dim_feedforward 2048 \
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
--num_epochs 20 \
--learning_rate 1e-4 \
--beta1 0.9 \
--beta2 0.98 \
--epsilon 1e-9 \
--schedule 'linear' \
--warmup_steps 100 \
--save_steps 100 \
--test_steps 100 \
--epsilon_ls 0.1 \
\
--do_evaluate \
--do_test \
--evaluate_batch_size 32 \
--num_beams 5 \
--alpha 0.6

torchrun  --nproc_per_node=4   main.py \
--datasets "Multi30k" \
--log_dir "./log_dir" \
--model "Transformer" \
--task "multi30k_de_en" \
--datasets "Multi30k" \
--data_dir "./data" \
--src_language "de" \
--tgt_language "en" \
--data_processors 10 \
\
--d_model 512 \
--num_heads 8 \
--dim_feedforward 2048 \
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
--num_epochs 20 \
--learning_rate 1e-4 \
--beta1 0.9 \
--beta2 0.98 \
--epsilon 1e-9 \
--schedule 'linear' \
--warmup_steps 100 \
--save_steps 100 \
--test_steps 100 \
--epsilon_ls 0.1 \
\
--do_evaluate \
--do_test \
--evaluate_batch_size 32 \
--num_beams 5 \
--alpha 0.6