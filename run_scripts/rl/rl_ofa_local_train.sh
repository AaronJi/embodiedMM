#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1052
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

env=hopper
dataset=medium-replay   #medium-replay  # expert # medium
data_dir=../../dataset/gym_data
data=${data_dir}/${env}-${dataset}-v2.tsv,${data_dir}/${env}-${dataset}-v2.tsv
restore_file=./checkpoints/ofa_base.pt
selected_cols=0,1,2,3,4,5,6,7
# save_dir=/data/output1/ofa_workspace/checkpoints/train_for_${task}_hopper_w_pretrain

task=mujoco_control_task
arch=ofa_traj_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.0
lr=1e-4
max_epoch=1
warmup_ratio=0.01
batch_size=32
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=800
max_tgt_length=300
num_bins=1000
patch_image_size=384
sample_patch_num=196
max_image_size=512

log_dir=./logs
save_dir=./checkpoints
mkdir -p $log_dir $save_dir
log_file=${log_dir}/${env}"_"${dataset}"_"${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}".log"
save_path=${save_dir}/${env}"_"${dataset}"_"${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}

python ../../train.py \
  $data \
  --selected-cols=${selected_cols} \
  --bpe-dir=${bpe_dir} \
  --user-dir=${user_dir} \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir=${save_path} \
  --task=${task} \
  --arch=${arch} \
  --criterion=${criterion} \
  --label-smoothing=${label_smoothing} \
  --batch-size=${batch_size} \
  --update-freq=${update_freq} \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --share-decoder-input-output-embed \
  --share-all-embeddings \
  --layernorm-embedding \
  --patch-layernorm-embedding \
  --code-layernorm-embedding \
  --resnet-drop-path-rate=${resnet_drop_path_rate} \
  --encoder-drop-path-rate=${encoder_drop_path_rate} \
  --decoder-drop-path-rate=${decoder_drop_path_rate} \
  --dropout=${dropout} \
  --attention-dropout=${attention_dropout} \
  --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=5.0 \
  --lr-scheduler=polynomial_decay --lr=${lr} \
  --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
  --log-format=simple --log-interval=10 \
  --fixed-validation-seed=7 \
  --keep-last-epochs=15 \
  --save-interval=1 \
  --save-interval-updates=6000 \
  --disable-validation \
  --max-src-length=${max_src_length} \
  --max-tgt-length=${max_tgt_length} \
  --add-type-embedding \
  --scale-attn \
  --scale-fc \
  --scale-heads \
  --disable-entangle \
  --num-bins=${num_bins} \
  --patch-image-size=${patch_image_size} \
  --sample-patch-num=${sample_patch_num} \
  --max-image-size=${max_image_size} \
  --cpu \
  --ddp-backend=no_c10d \
  --use-bmuf \
  --fp16-scale-window=128 \
  --num-workers=0 > ${log_file} 2>&1

# --restore-file=${restore_file} \
# --freeze-encoder-embedding \
# --freeze-decoder-embedding \
# --ddp-backend=no_c10d \
# --use-bmuf \
# --drop-worst-ratio=${drop_worst_ratio} \
# --drop-worst-after=${drop_worst_after} \