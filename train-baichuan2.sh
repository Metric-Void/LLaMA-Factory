CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/allen/Codes/models/Baichuan2-13B-Chat \
    --do_train \
    --dataset metricsubs \
    --template baichuan2 \
    --finetuning_type lora \
    --lora_target W_pack \
    --output_dir /home/allen/Codes/models/baichuan2-chat-finetuned-lora \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 5 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --quantization_bit 8
    # --system_prompt "你是一个擅长翻译科技新闻的翻译专家。请将以下内容翻译为中文，使用相同格式输出，并保留时间戳。不要漏掉任何信息。合并多行文本时，保留第一个和最后一个时间戳。" \