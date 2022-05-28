
import argparse
import logging
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("embodied_rl.train")

from trainer_light import TrainerLight
from tasks import *

def main(cfg: argparse.Namespace) -> None:

    logger.info(args)

    task = MujocoControlTask(cfg)
    task.build_env()
    task.load_dataset()

    assert cfg.criterion, "Please specify criterion to train a model"

    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)

    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))

    quantizer = None
    trainer = TrainerLight(cfg, task, model, criterion, quantizer)
    trainer._build_optimizer()

    train(cfg, trainer)
    return

def train(cfg, trainer):
    max_epoch = cfg.max_iters

    for iter in range(max_epoch):
        outputs = trainer.train_iteration(num_steps=cfg.num_steps_per_iter, iter_num=iter + 1, print_logs=True)

    return

'''
    exit(0)
    print(cfg.common.tpu)
    print(cfg.dataset.train_subset)
    print(max_epoch)
    print(epoch_itr)
    print(len(epoch_itr))
    print(cfg.optimization.update_freq)
    print(math.ceil(len(epoch_itr) / cfg.optimization.update_freq[0]))
    print(total_num_updates)
    print(trainer.get_num_updates())
    print('#'*10)

    task.load_dataset('train', combine=False, epoch=1)
    #print(task.datasets['valid'])
    #dataset_form = 'valid'
    dataset_form = 'train'
    unify_dataset = task.datasets[dataset_form]

    index = 0

    pair_samples = unify_dataset.process_image_text_pair(index)
    #print_list_of_dict(pair_samples)
    print('#'*10)
    pure_text_examples = unify_dataset.process_pure_text(0)
    #print_list_of_dict(pure_text_examples)
    #print('#' * 10)
    pure_image_examples = unify_dataset.process_pure_image(0)
    #print_list_of_dict(pure_image_examples)
    #print('#' * 10)
    #pure_detection_examples = unify_dataset.process_detection(0)
    #print_list_of_dict(pure_detection_examples)
    #print('#' * 10)


    import base64
    from PIL import Image
    from io import BytesIO


    list_v_rel = torch.tensor([1.0, 0.1, 0.005, 0.95, 0, 0.87])
    q_item = quantize(list_v_rel, unify_dataset.num_bins, unify_dataset.encode_text)
    print(q_item)

    #print(unify_dataset.src_dict.pad())  # 1
    #print(unify_dataset.bos)  # 0
    #print(unify_dataset.eos)  # 2

    print('#' * 10)
    #samples = [unify_dataset.process_image_text_pair(index) for index in [0, 3, 10]]
    pair_samples, extra_samples = unify_dataset[index]
    print(type(pair_samples))
    #print(pair_samples)
    print(len(pair_samples))
    print(type(extra_samples))
    #print(extra_samples)
    print(len(extra_samples))
    print('#' * 10)
    #pair_samples1, extra_samples1 = unify_dataset[index+5]
    #samples = pair_samples + pair_samples1
    #from data.pretrain_data.unify_dataset import collate
    #batch = collate(samples, unify_dataset.src_dict.pad(), unify_dataset.eos)
    #print_batch(batch)

    samples = [unify_dataset[index], unify_dataset[index+5]]
    res_v1, res_v2 = unify_dataset.collater(samples)

    print_batch(res_v1)
    print('#' * 10)
    print_batch(res_v2)

q_tokens = []
for v_rel in tensor_v_rel:
    assert 0 <= v_rel <= 1
    print((v_rel * (num_bins - 1)))
    bin = int((v_rel * (num_bins - 1)).round())
    q_token = "<bin_{}>".format(bin)
    print(bin, q_token)
    q_tokens.append(q_token)
print(q_tokens)
'''


def quantize(tensor_v_rel, num_bins, encode_fun):
    q_tokens = ["<bin_{}>".format(int((v_rel * (num_bins - 1)).round())) for v_rel in tensor_v_rel]
    q_item = encode_fun(' '.join(q_tokens), use_bpe=False)
    return q_item


def print_batch(batch):
    #print(batch)
    for k in batch:
        print(k)
        if k == 'net_input':
            for kk in batch[k]:
                print('  ', kk, batch[k][kk].shape if batch[k][kk] is not None else None)
        elif k == 'target':
            print(batch[k].shape)
        else:
            print(batch[k])
    print(batch['net_input']['src_lengths'])

    src_lengths = batch['net_input']['src_lengths']
    print(src_lengths)
    print(src_lengths.sum())
    print(src_lengths.sum().item())
    return

def print_list_of_dict(list_dict):
    for ps in list_dict:
        print(type(ps))
        for pps in ps:
            print(pps)
            print(ps[pps])

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')  # hopper, walker2d, halfcheetah
    parser.add_argument('--dataset', type=str, default='medium-replay')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning, ofa for OFA
    parser.add_argument('--criterion', type=str, default='action_pred_error')  # action_pred_error, label_smoothed_cross_entropy
    parser.add_argument('--num_bins', type=int, default=1000)
    parser.add_argument('--action_pred_way', type=str, default='max')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--bpe_dir', type=str, default='utils/BPE')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    args = parser.parse_args()

    main(args)
