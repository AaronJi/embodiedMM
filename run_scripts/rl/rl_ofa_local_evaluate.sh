#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1081
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

env=hopper
dataset=expert   #medium-replay  # expert # medium
data_dir=../../dataset/gym_data
data=${data_dir}/${env}-${dataset}-v2.tsv

save_dir=./checkpoints
max_epoch=1
warmup_ratio=0.01
drop_worst_after=5000
path=../../run_scripts/rl/checkpoints/checkpoint_last.pt
path=${save_dir}/${env}"_"${dataset}"_"${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}/checkpoint_last.pt

result_path=../../results/rl
selected_cols=1,4,2
split='test'
task=mujoco_control_task
criterion=adjust_label_smoothed_cross_entropy

#python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate_interactive.py \
python ../../evaluate_interactive.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=${task} \
    --criterion=${criterion} \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --cpu \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"
