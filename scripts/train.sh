# mode, model, dataset
model_type=$2
dataset=$4
suffix=$6

# distributed training envs
WORLD_SIZE=$(nvidia-smi --list-gpus | wc -l)
# MASTER_PORT=$((10000 + $RANDOM % 10000))

# set up output directory
project_dir='.'
output_dir=${project_dir}/out/${dataset}/${model_type}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

# -m torch.distributed.run --nproc_per_node $WORLD_SIZE --master_port $MASTER_PORT \
# -m debugpy --listen 12347 --wait-for-client \
python \
    -m debugpy --listen 12347 --wait-for-client \
    -m torch.distributed.run --nproc_per_node $WORLD_SIZE --master_port 12346 \
    main.py\
    --mode train --output_dir $output_dir --ckpt_dir $ckpt_dir \
    $@ 2>&1 | tee ${output_dir}/log.txt

# deepspeed main.py \
#     --mode train --output_dir $output_dir --ckpt_dir $ckpt_dir --deepspeed ds_config.json \
#     $@ 2>&1 | tee ${output_dir}/log.txt

# accelerate launch main.py \
#     --mode train --output_dir $output_dir --ckpt_dir $ckpt_dir \
#     $@ 2>&1 | tee ${output_dir}/log.txt
