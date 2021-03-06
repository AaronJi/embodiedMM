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

from utils import checkpoint_utils
from utils.rl.rl_utils import reform_window_len

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

on_local = True

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
    if on_local:
        task.load_dataset('valid', task_cfg=saved_cfg.task)
    else:
        task.build_local_env()
        task.load_local_dataset('valid', file_path=cfg.task.data)

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

    #print(cfg.criterion._name)
    #criterion = task.build_criterion('adjust_label_smoothed_cross_entropy')
    criterion = task.build_criterion(saved_cfg.criterion)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)
    tokenizer = Tokenizer()

    num_eval_episodes = 30
    outputs_list = []

    env_targets = [3000]
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


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}

#from utils.eval_utils import decode_fn
def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x

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

def get_result(task, generator, models, sample, tokenizer, get_first_tokens=None):
    import string

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
    else:
        action_pred = result
    return torch.tensor(np.array(action_pred))

def get_ref_result(task, models, sample, tokenizer, criterion, get_first_tokens=None):
    model = models[0]

    net_output = model(**sample['net_input'])
    lprobs, target, constraint_masks = criterion.get_lprobs_and_target(model, net_output, sample)

    if on_local:
        base_index = 0
    else:
        base_index = len(task.tgt_dict) - task.datasets['valid'].num_bins  # 58457
    result = []

    mode = 'max'
    #mode = 'mean'
    if mode == 'max':
        if on_local:
            max_prob, max_i = torch.max(lprobs, dim=1)
        else:
            max_prob, max_i = torch.max(lprobs[:, -task.datasets['valid'].num_bins:], dim=1)
        for ii in max_i:
            try:
                vv = tokenizer.decode(task.tgt_dict[base_index + int(ii)])
                v = float(vv) / task.datasets['valid'].num_bins
            except:
                v = 0.0
            result.append(v)
    else:
        for i in range(lprobs.shape[0]):
            result_v = 0.0
            result_p = 0.0
            for j in range(task.datasets['valid'].num_bins):
                p = lprobs[i, base_index + j]
                try:
                    vv = tokenizer.decode(task.tgt_dict[base_index + int(j)])
                    v = float(vv) / task.datasets['valid'].num_bins
                except:
                    v = 0.0
                result_v += v*p
                result_p += p
            result_v = result_v / result_p
            result.append(result_v)

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


    gamma = 1.0
    #gamma = task.cfg.gamma

    #eval_steps = task.max_ep_len
    eval_steps = 5000

    # statistics
    state_mean = task.state_mean
    state_std = task.state_std
    action_mean = task.action_mean
    action_std = task.action_std
    state_lb = task.state_bounds[0]
    state_ub = task.state_bounds[1]
    action_lb = task.action_bounds[0]
    action_ub = task.action_bounds[1]

    state = task.env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    padding_to_full_window = True
    pred_is_cont = True
    generate_action_once = True

    if padding_to_full_window:
        state_padding_num = task.cfg.state_padding_num
        action_padding_num = task.cfg.action_padding_num
        return_padding_num = 1.0
    else:
        state_padding_num = None
        action_padding_num = None
        return_padding_num = None

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = state.reshape(1, task.state_dim)
    actions = np.zeros((0, task.action_dim))
    rewards = np.zeros((0, 1))
    returns = target_return * np.ones((1, 1))
    timesteps = np.zeros((1, 1))
    masks = np.zeros((1, 1))

    ep_return = target_return
    episode_return, episode_length = 0, 0

    def scale_vars(states, actions, returns):
        if task.cfg.scale_way == 'normalize':
            states_rel = (states - state_mean)/state_std
            actions_rel = (actions - action_mean) / action_std
        else:
            states_rel = (states - state_lb)/(state_ub - state_lb)  # .detach().cpu().numpy()
            actions_rel = (actions - action_lb) / (action_ub - action_lb)
        returns_rel = returns/ep_return

        return states_rel, actions_rel, returns_rel

    def rescale_vars(action_rel):
        if task.cfg.scale_way == 'normalize':
            action = action_rel*action_std + action_mean
        else:
            action = action_lb + action_rel * (action_ub - action_lb)

        return action

    for t in range(eval_steps):
        id = task.env.name + '-t=' + str(t) + '-eval'
        if t % 10 == 0:
            print('t=%i $$$$$$$$$$$' % t)

        # add padding
        #actions = torch.cat([actions, 0 * torch.ones((1, task.env.action_dim), device=device)], dim=0)
        actions = np.concatenate([actions, action_padding_num * np.ones((1, task.env.action_dim))], axis=0)

        # get recent window
        window_states = reform_window_len(states, task.cfg.window_len, padding_num=state_padding_num)
        window_actions = reform_window_len(actions, task.cfg.window_len, padding_num=action_padding_num)
        window_returns = reform_window_len(returns, task.cfg.window_len, padding_num=0)  # returns[0, 0]
        window_rewards = reform_window_len(rewards, task.cfg.window_len, padding_num=0)
        window_timesteps = reform_window_len(timesteps, task.cfg.window_len, padding_num=0)
        window_masks = reform_window_len(masks, task.cfg.window_len, padding_num=1).reshape(-1)

        window_states_rel, window_actions_rel, window_returns_rel = scale_vars(window_states, window_actions, window_returns)

        if pred_is_cont:
            uniq_id, src, tgt, timesteps, mask, prev_tgt = task.datasets['valid'].process_trajectory_from_vars(id, torch.tensor(window_states_rel), torch.tensor(window_actions_rel), torch.tensor(window_rewards), None, torch.tensor(window_returns_rel), torch.tensor(window_timesteps), torch.tensor(window_masks), inference=True)
            example = task.datasets['valid'].process_train_seq(uniq_id, src, tgt, timesteps, mask, prev_tgt)
            sample = task.datasets['valid'].collater([example])

            #from data.rl_data.gym_dataset import print_batch
            #print_batch(sample)

            net_output = models[0](**sample['net_input'])

            action_pred_rel = net_output[0][0][-1][-task.env.action_dim:]
            action_pred_rel = action_pred_rel.detach().cpu().numpy()
        else:
            if generate_action_once:
                uniq_id, src, tgt, timesteps, mask = task.datasets['valid'].process_trajectory_from_vars(id, torch.tensor(window_states_rel), torch.tensor(window_actions_rel), torch.tensor(window_rewards), None, torch.tensor(window_returns_rel), torch.tensor(window_timesteps), torch.tensor(window_masks), inference=True)
                example = task.datasets['valid'].process_train_seq(uniq_id, src, tgt, timesteps, mask)
                sample = task.datasets['valid'].collater([example])

                #result = get_result(task, generator, models, sample, tokenizer)
                result = get_ref_result(task, models, sample, tokenizer, criterion)
                action_pred_rel = result.detach().cpu().numpy()
            else:
                action_pred_rel = []
                for tai in range(task.action_dim):
                    print('^'*20)
                    print(tai)
                    if tai == 0:
                        rsa = torch.cat([torch.tensor(window_returns_rel).reshape(-1, 1), torch.tensor(window_states_rel).reshape(-1, task.state_dim), torch.tensor(window_actions_rel).reshape(-1, task.action_dim)], dim=-1).reshape(-1)
                        src = rsa[:-task.action_dim]
                        tgt = torch.ones((0))
                    else:
                        tgt = torch.cat([tgt, a_pred_rel.reshape(-1)], dim=0)

                    example = task.datasets['valid'].process_token_seq(id, src, tgt)
                    sample = task.datasets['valid'].collater([example])
                    #result = get_result(task, generator, models, sample, tokenizer)
                    result = get_ref_result(task, models, sample, tokenizer, criterion)
                    print(src)
                    print(tgt)
                    print(sample['net_input']['src_tokens'])
                    print(sample['net_input']['prev_output_tokens'])
                    print(sample['target'])
                    print(result)
                    a_pred_rel = result[-1]
                    action_pred_rel.append(a_pred_rel)

                action_pred_rel = np.array(action_pred_rel)

        action_pred = rescale_vars(action_pred_rel)
        #print(action_pred_rel, action_pred)
        action = action_pred
        actions[-1] = action

        state, reward, done, _ = task.env.step(action)
        target_return = (target_return - reward) / gamma

        states = np.concatenate([states, state.reshape((-1, task.state_dim))])
        rewards = np.concatenate([rewards, np.array(rewards).reshape((-1, 1))])
        returns = np.concatenate([returns, np.array(target_return).reshape((-1, 1))])
        timesteps = np.concatenate([timesteps, (t + 1)*np.ones((1, 1))])
        masks = np.concatenate([masks, np.zeros((1, 1))])

        episode_return += reward
        episode_length += 1

        if done:
            print('episode lasts %i steps before failing' % (t+1))
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
