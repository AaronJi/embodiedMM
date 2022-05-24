
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')  # hopper, walker2d, halfcheetah
    parser.add_argument('--dataset', type=str, default='medium-replay')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--criterion', type=str, default='action_pred_error')  # action_pred_error, label_smoothed_cross_entropy
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
