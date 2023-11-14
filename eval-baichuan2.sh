CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name_or_path /home/allen/Codes/models/Baichuan2-13B-Chat \
    --finetuning_type lora \
    --checkpoint_dir /home/allen/Documents/models/baichuan2-chat-finetuned-lora \
    --template baichuan2 \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 1 \
    --quantization_bit 8 \
    --per_device_eval_batch_size 1



CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/allen/Codes/models/Baichuan2-13B-Chat \
    --checkpoint_dir /home/allen/Documents/models/baichuan2-chat-finetuned-lora \
    --predict_with_generate True \
    --finetuning_type lora \
    --quantization_bit 8 \
    --template baichuan2 \
    --dataset_dir data \
    --dataset metricsubs \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 1024 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/Custom/lora/eval_base \
    --do_predict True \
    --split test 

