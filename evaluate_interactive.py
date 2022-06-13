#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
import os
import sys

import numpy as np
import torch
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.utils import reset_logging
from omegaconf import DictConfig

from utils import checkpoint_utils, checkpoint_utils
from utils.eval_utils import eval_step, merge_results

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def main(cfg: DictConfig, **kwargs):
    utils.import_user_module(cfg.common)

    reset_logging()
    logger.info(cfg)

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Load ensemble
    overrides = eval(cfg.common_eval.model_overrides)
    # Deal with beam-search / all-candidate VQA eval
    if cfg.task._name == "vqa_gen":
        overrides['val_inference_type'] = "beamsearch" if kwargs['beam_search_vqa_eval'] else "allcand"

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))

    if kwargs["zero_shot"]:
        task = tasks.setup_task(cfg.task)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
    else:
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset('valid', task_cfg=saved_cfg.task)

    # Move models to GPU
    for model, ckpt_path in zip(models, utils.split_paths(cfg.common_eval.path)):
        if kwargs['ema_eval']:
            logger.info("loading EMA weights from {}".format(ckpt_path))
            model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    device = 'cpu'
    model.to(device=device)

    criterion = task.build_criterion('adjust_label_smoothed_cross_entropy')

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)
    tokenizer = Tokenizer()

    env_targets = [3000]

    num_eval_episodes = 10
    outputs_list = []
    for tar in env_targets:
        returns, lengths = [], []
        for i in range(num_eval_episodes):
            print('eval target=%i, i_episode=%i' %(tar, i))
            with torch.no_grad():
                ret, length = evaluate_episode_rtg(
                    task,
                    models,
                    generator,
                    tokenizer,
                    criterion,
                    device='cpu',
                    target_return=tar,  #  / task.scale
                    mode='normal',
                )
        returns.append(ret)
        lengths.append(length)

        output = {
            f'target_{tar}_return_mean': np.mean(returns),
            f'target_{tar}_return_std': np.std(returns),
            f'target_{tar}_length_mean': np.mean(lengths),
            f'target_{tar}_length_std': np.std(lengths),
        }
        outputs_list.append(output)

    logs = dict()
    for outputs in outputs_list:
        for k, v in outputs.items():
            logs[f'evaluation/{k}'] = v

    print('=' * 80)
    for k, v in logs.items():
        print(f'{k}: {v}')


    '''
    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    results = []
    score_sum = torch.FloatTensor([0]).cuda()
    score_cnt = torch.FloatTensor([0]).cuda()
    for sample in progress:
        if "net_input" not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if cfg.common.fp16 else sample
        with torch.no_grad():
            result, scores = eval_step(task, generator, models, sample, **kwargs)
        results += result
        score_sum += sum(scores) if scores is not None else 0
        score_cnt += len(scores) if scores is not None else 0
        progress.log({"sentences": sample["nsentences"]})

    merge_results(task, cfg, logger, score_cnt, score_sum, results)    
    '''

    return


def padding_to_window(win_data, win_len, padding_num=None, batch_mode=False):

    if padding_num is None:
        padding_num = 0

    if batch_mode:
        tlen = win_data.shape[1]
        dim = win_data.shape[2]
        dummy_array = np.ones((1, win_len - tlen, dim))
        cat_dim = 1
    else:
        tlen = win_data.shape[0]
        dim = win_data.shape[1]
        dummy_array = np.ones((win_len - tlen, dim))
        cat_dim = 0

    win_data = np.concatenate([padding_num * dummy_array, win_data],  axis=cat_dim)
    return win_data

from utils.eval_utils import decode_fn
class Tokenizer(object):
    def __init__(self):
        return

    def encode(self, v):
        x = v
        return x

    def decode(self, x):
        vv = x.split(' ')
        v = []
        for vvv in vv:
            try:
                vvvv = vvv[1:-1].split('_')[1]
            except:
                vvvv = 'None'
            v.append(vvvv)
        return ' '.join(v)

import string
def get_result(task, generator, models, sample, tokenizer, get_first_tokens=None):
    hypos = task.inference_step(generator, models, sample)
    i = 0
    detok_hypo_str = decode_fn(hypos[i][0]["tokens"], task.tgt_dict, None, generator, tokenizer=tokenizer)
    transtab = str.maketrans({key: None for key in string.punctuation})
    bins = detok_hypo_str.translate(transtab).strip().split(' ')
    result = []
    for b in bins:
        try:
            v = float(b) / task.datasets['valid'].num_bins
        except:
            v = 0.0
        result.append(v)
    if get_first_tokens is not None:
        action_pred = result[:get_first_tokens]
    return torch.tensor(np.array(action_pred))

def get_ref_result(task, models, sample, tokenizer, criterion, get_first_tokens=None):
    model = models[0]
    #print(sample)
    net_output = model(**sample['net_input'])
    #print(net_output)

    lprobs, target, constraint_masks = criterion.get_lprobs_and_target(model, net_output, sample)
    max_prob, max_i = torch.max(lprobs, dim=1)

    result = []
    for ii in max_i:
        #print(task.tgt_dict[int(ii)])
        #print(tokenizer.decode(task.tgt_dict[int(ii)]))
        try:
            vv = tokenizer.decode(task.tgt_dict[int(ii)])
            v = float(vv) / task.datasets['valid'].num_bins
        except:
            v = 0.0
        result.append(v)

    #print('result is %s' %str(result))
    if get_first_tokens is not None:
        action_pred = result[:get_first_tokens]
    else:
        action_pred = result
    return torch.tensor(np.array(action_pred))


def evaluate_episode_rtg(
        task,
        models,
        generator,
        tokenizer,
        criterion,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    #max_ep_len = 1000,
    #scale = 1000.,

    state_lb = torch.from_numpy(task.state_bounds[0]).to(device=device)
    state_ub = torch.from_numpy(task.state_bounds[1]).to(device=device)
    action_lb = torch.from_numpy(task.action_bounds[0]).to(device=device)
    action_ub = torch.from_numpy(task.action_bounds[1]).to(device=device)

    state = task.env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    padding_to_full_window = True
    generate_action_once = False

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, task.env.state_dim).to(device=device, dtype=torch.float32)
    actions = 0*torch.ones((0, task.env.action_dim), device=device, dtype=torch.float32)
    rewards = 0*torch.ones(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0

    for t in range(task.max_ep_len):
        id = task.env.name + '-t=' + str(t) + '-eval'

        # add padding
        actions = torch.cat([actions, 0*torch.ones((1, task.env.action_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, 0*torch.ones(1, device=device)])

        states_rel = (states - state_lb)/(state_ub - state_lb)
        actions_rel = (actions - action_lb) / (action_ub - action_lb)
        rtg_rel = target_return/ep_return

        # get recent window
        states_rel = states_rel[-task.cfg.window_len:, :]
        actions_rel = actions_rel[-task.cfg.window_len:, :]
        rtg_rel = rtg_rel[-task.cfg.window_len:, :]

        # padding
        if padding_to_full_window:
            states_rel = padding_to_window(states_rel, task.cfg.window_len, padding_num=task.cfg.state_padding_num)
            actions_rel = padding_to_window(actions_rel, task.cfg.window_len, padding_num=task.cfg.action_padding_num)
            rtg_rel = padding_to_window(rtg_rel, task.cfg.window_len, padding_num=1.0)

        if generate_action_once:
            example = task.datasets['valid'].process_trajectory_from_vars(id, states_rel, actions_rel, rtg_rel, inference=True)
            sample = task.datasets['valid'].collater([example])

            action_pred_rel = get_result(task, generator, models, sample, tokenizer, get_first_tokens=3)
            # action_pred_rel = get_ref_result(task, models, sample, tokenizer, criterion, get_first_tokens=1)
        else:
            action_pred_rel = []
            for tai in range(3):
                if tai == 0:
                    ss = torch.tensor(states_rel)
                    aa = torch.tensor(actions_rel)
                    rtgg = torch.tensor(rtg_rel)
                    r_s_a = torch.cat([rtgg.reshape(-1, 1), ss.reshape(-1, 11), aa.reshape(-1, 3)], dim=-1).reshape(-1)
                    src = r_s_a[:-3]
                    #tgt = r_s_a[-3:]
                    tgt = torch.ones((0))
                else:
                    tgt = torch.cat([tgt, a_pred_rel.reshape(-1)], dim=0)


                example = task.datasets['valid'].process_token_seq(id, src, tgt)

                #example = task.datasets['valid'].process_trajectory_from_vars(id, states_rel, actions_rel, rtg_rel, inference=True)
                sample = task.datasets['valid'].collater([example])
                #action_pred_rel = get_result(task, generator, models, sample, tokenizer, get_first_tokens=1)

                result = get_ref_result(task, models, sample, tokenizer, criterion)
                a_pred_rel = result[-1]


                action_pred_rel.append(a_pred_rel)

            action_pred_rel = torch.tensor(np.array(action_pred_rel))

        action_pred = action_lb + action_pred_rel*(action_ub - action_lb)
        actions[-1] = action_pred


        action = action_pred.detach().cpu().numpy()

        state, reward, done, _ = task.env.step(action)
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, task.env.state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - reward
        else:
            pred_return = target_return[0,-1]

        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=0)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--ema-eval", action='store_true', help="Use EMA weights to make evaluation.")
    parser.add_argument("--beam-search-vqa-eval", action='store_true', help="Use beam search for vqa evaluation (faster inference speed but sub-optimal result), if not specified, we compute scores for each answer in the candidate set, which is slower but can obtain best result.")
    parser.add_argument("--zero-shot", action='store_true')
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(
        cfg, main, ema_eval=args.ema_eval, beam_search_vqa_eval=args.beam_search_vqa_eval, zero_shot=args.zero_shot
    )


if __name__ == "__main__":
    cli_main()
