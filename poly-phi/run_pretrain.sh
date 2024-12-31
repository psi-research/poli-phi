#deepspeed --num_gpus=4 --num_nodes 1 run_pretrain.py --resume_from_checkpoint "./output_ds/checkpoint-10" --deepspeed "./config/zero_stage2_config.json"
#deepspeed --include localhost:0 run_pretrain.py --deepspeed "./config/zero_stage2_config.json"
#deepspeed --num_gpus=4 --num_nodes 1 run_pretrain.py --resume_from_checkpoint "./output_ds/checkpoint-40" --deepspeed "./config/zero_stage2_config.json"

export TOKENIZERS_PARALLELISM=false

deepspeed --num_gpus=8 --num_nodes 1  run_pretrain.py \
    --model_name_or_path phi-3_deepspeed_mini_130g_4k_2k \
    --model_dir ./model \
    --model_type mini \
    --dataset_name_or_path ./data/tokenized_dataset_130g_2k_v4_R2 \
    --tokenizer_name_or_path ./tokenizer/tokenizer_v7 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --output_dir ./output \
    --context_length 2048 \
    --eval_steps 100 \
    --logging_steps 100 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --max_steps 50000 \
    --learning_rate 1.5e-4 \
    --save_steps 500 \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --save_total_limit 5 \
    --weight_decay 0.1 \
    --warmup_steps 5000 \
    --deepspeed ./config/zero_stage2_config.json
