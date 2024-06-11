#!/bin/bash
dataset=ogbn-arxiv

model_type=e5-revgat
suffix=main

output_dir=out/${dataset}/${model_type}/${suffix}
ckpt_dir=${output_dir}/ckpt

# mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

python "main_.py" 2>&1 | tee "${output_dir}/log.txt"