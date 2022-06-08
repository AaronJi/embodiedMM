#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1081
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

env=hopper
dataset=medium-replay
rl_data_dir=../../dataset/gym_data
data=${rl_data_dir}/${env}-${dataset}-v2.pkl

#data=../../dataset/caption_data/caption_test.tsv
path=../../run_scripts/rl/checkpoints_cloud/checkpoint_last.pt
result_path=../../results/rl
selected_cols=1,4,2
split='test'
task=mujoco_control_task

#python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate_interactive.py \
python ../../evaluate_interactive.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=${task} \
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
