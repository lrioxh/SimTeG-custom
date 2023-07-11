dataset=ogbl-citation2
model_type=roberta-large
suffix=fix

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo roberta-large \
    --eval_batch_size 200 \
    --mode "test" \
    --deepspeed ds_config.json \
    --fp16
