dataset=ogbn-arxiv
model_type=GAMLP

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_${lm_model_type}_fix
bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt
bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.37 \
    --gnn_label_smoothing 0.1 \
    --gnn_lr 0.007 \
    --gnn_num_layers 5 \
    --gnn_warmup_ratio 0.41 \
    --gnn_weight_decay 5e-6 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

lm_model_type=all-roberta-large-v1
suffix=main_X_${lm_model_type}_fix
bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt
bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.15 \
    --gnn_label_smoothing 0.47 \
    --gnn_lr 0.001 \
    --gnn_num_layers 6 \
    --gnn_warmup_ratio 0.2 \
    --gnn_weight_decay 8.5e-5 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

lm_model_type=e5-large
suffix=main_X_${lm_model_type}_fix
bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt
bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.33 \
    --gnn_label_smoothing 0.18 \
    --gnn_lr 0.002 \
    --gnn_num_layers 4 \
    --gnn_warmup_ratio 0.34 \
    --gnn_weight_decay 9e-7 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir
