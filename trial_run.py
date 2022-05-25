#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
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
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
from fairseq import (
    # checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
# from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

from utils import checkpoint_utils
from trainer import Trainer

def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    # data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    print(cfg.dataset.combine_valid_subsets)

    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        print(cfg.dataset.valid_subset)
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

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
    q_item = quantizer(list_v_rel, unify_dataset.num_bins, unify_dataset.encode_text)
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
    return

def quantizer(tensor_v_rel, num_bins, encode_fun):
    '''
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

def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:

    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()

    return

if __name__ == "__main__":
    cli_main()
