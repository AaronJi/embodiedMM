# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import os
import numpy as np
from typing import Optional
from omegaconf import DictConfig
from fairseq.tasks import register_task
from fairseq.data import Dictionary

from tasks.ofa_task import OFATask, OFAConfig
from data.rl_data.gym_dataset import GymDataset
from data.file_dataset import FileDataset
from utils.statistic_utils import cal_stat
from utils.rl.rl_utils import discount_cumsum, get_str_from_1darray

logger = logging.getLogger(__name__)
on_cloud = False  # if is on the cloud-computing, i.e. pai & odps

@dataclass
class MujocoControlConfig(OFAConfig):
    max_image_size: int = field(
        default=512, metadata={"help": ""}
    )
    text_data: Optional[str] = field(
        default=None,
        metadata={"help": "pure text data"},
    )
    image_data: Optional[str] = field(
        default=None,
        metadata={"help": "pure image data"},
    )
    detection_data: Optional[str] = field(
        default=None,
        metadata={"help": "detection data"},
    )
    text_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "pure text data selected cols"},
    )
    image_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "pure image data selected cols"},
    )
    detection_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "detection data selected cols"},
    )
    neg_sample_dir: Optional[str] = field(
        default=None,
        metadata={"help": "negative sample directory, which contains captions (taken from all image-text pairs), "
                          "answers (taken from VQA), "
                          "objects (taken form OpenImages) "},
    )
    code_image_size: int = field(
        default=128, metadata={"help": "the resolution of the generated image in the image infilling task"}
    )

    pretrain_seed: int = field(
        default=7,
        metadata={"help": "pretrain seed"},
    )

    mask_ratio: float = field(
        default=0.3,
        metadata={"help": "fraction of words/subwords that will be masked"},
    )
    random_ratio: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], use random token this often"},
    )
    keep_ratio: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], keep original token this often"},
    )
    mask_length: str = field(
        default="span-poisson",
        metadata={"help": "mask length to choose ['subword', 'word', 'span-poisson']"},
    )
    poisson_lambda: float = field(
        default=3.0,
        metadata={"help": "randomly shuffle sentences for this proportion of inputs"},
    )
    replace_length: int = field(
        default=1,
        metadata={"help": "when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)"},
    )
    reward_mode: str = field(
        default='normal',
        metadata={"help": "mode of reward generaltion, normal or sparsed"},
    )
    pct_traj: float = field(
        default=1.0,
        metadata={"help": "percentage of sampled top-performed trajectories"},
    )
    max_len: int = field(
        default=20,
        metadata={"help": "window length of sampled trajectory data"},
    )
    extract_way: str = field(
        default='all',
        metadata={"help": "way of window generation: all, full or rand"},
    )
    scale_way: str = field(
        default='minmax',
        metadata={"help": "scale method of state and action: minmax or normalize"},
    )


@register_task("mujoco_control_task", dataclass=MujocoControlConfig)
class MujocoControlTask(OFATask):
    def __init__(self, cfg: MujocoControlConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.build_env()
        if not on_cloud:
            self.load_traj_spec()
        self.separator = "\t"

        return

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""

        src_dict = Dictionary()
        tgt_dict = Dictionary()
        # load dictionaries
        #src_dict = cls.load_dictionary(os.path.join(cfg.bpe_dir, "dict.txt"))
        #tgt_dict = cls.load_dictionary(os.path.join(cfg.bpe_dir, "dict.txt"))
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        #for i in range(cfg.code_dict_size):
        #    src_dict.add_symbol("<code_{}>".format(i))
        #    tgt_dict.add_symbol("<code_{}>".format(i))
        # quantization
        for i in range(cfg.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))

        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))
        return cls(cfg, src_dict, tgt_dict)


    def build_env(self):
        env = 'hopper'
        self.model = 'ofa'

        if env == 'hopper':
            self.env_name = 'Hopper-v3'
            self.max_ep_len = 1000
            self.env_targets = [3600, 1800]  # evaluation conditioning targets
            self.scale = 1000.  # normalization for rewards/returns
        elif env == 'halfcheetah':
            self.env_name = 'HalfCheetah-v3'
            self.max_ep_len = 1000
            self.env_targets = [12000, 6000]
            self.scale = 1000.
        elif env == 'walker2d':
            self.env_name = 'Walker2d-v3'
            self.max_ep_len = 1000
            self.env_targets = [5000, 2500]
            self.scale = 1000.
        elif env == 'reacher2d':
            self.env_name = 'Reacher2d'
            self.max_ep_len = 100
            self.env_targets = [76, 40]
            self.scale = 10.
        else:
            raise NotImplementedError

        if self.model == 'bc':
            self.env_targets = self.env_targets[:1]  # since BC ignores target, no need for different evaluations

        if on_cloud:
            from environments.rl_environments.gym_environment import GymEnvironment
            self.env = GymEnvironment(None, self.env_name)
            self.state_dim = self.env.state_dim
            self.act_dim = self.env.act_dim
        else:
            self.state_dim = 11
            self.act_dim = 3

        return

    def load_traj_spec(self):
        gym_file_path = '../../dataset/gym_data/hopper-medium-replay-v2.pkl'

        import pickle
        with open(gym_file_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        states, actions, rewards, returns, traj_lens = [], [], [], [], []
        for path in self.trajectories:
            if self.cfg.reward_mode == 'delayed':  # delayed: all rewards moved to end of trajectory
                path['rewards'][-1] = path['rewards'].sum()
                path['rewards'][:-1] = 0.
            states.append(path['observations'])
            actions.append(path['actions'])
            rewards.append(path['rewards'])
            returns.append(path['rewards'].sum())
            traj_lens.append(len(path['observations']))

        self.num_timesteps = sum(traj_lens)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)

        #state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        #self.state_mean = state_mean
        #self.state_std = state_std

        self.states = states
        self.actions = actions
        self.rewards = rewards

        self.state_mean, self.state_std, self.state_bounds, self.state_norm_bounds = cal_stat(states)
        self.action_mean, self.action_std, self.action_bounds, self.action_norm_bounds = cal_stat(actions)
        self.reward_mean, self.reward_std, self.reward_bounds, self.reward_norm_bounds = cal_stat(rewards)
        self.return_mean, self.return_std, self.return_bounds, self.return_norm_bounds = cal_stat(returns)

        traj_lens, returns = np.array(traj_lens), np.array(returns)
        self.traj_lens = traj_lens
        self.returns = returns

        pct_traj = self.cfg.pct_traj

        # only train on top pct_traj trajectories (for %BC experiment)
        self.num_timesteps = max(int(pct_traj * self.num_timesteps), 1)
        sorted_inds = np.argsort(returns)  # lowest to highest
        self.num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= self.num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            self.num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-self.num_trajectories:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        self.p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
        return

    def convert_pickle_to_tsv(self, file_path, split):

        gym_data = self.extract_trajectories(split)
        print('writing to tsv file %s...' % file_path)

        gym_split_file = open(file_path, 'w')
        for index in range(len(gym_data)):
            uniq_id, s, a, r, d, rtg, timesteps, mask = gym_data[index]
            line = uniq_id + self.separator + get_str_from_1darray(s.reshape(-1)) + self.separator + get_str_from_1darray(a.reshape(-1)) + self.separator + get_str_from_1darray(
                r.reshape(-1)) + self.separator + get_str_from_1darray(d.reshape(-1)) + self.separator + get_str_from_1darray(rtg.reshape(-1)) + self.separator + get_str_from_1darray(
                timesteps.reshape(-1)) + self.separator + get_str_from_1darray(mask.reshape(-1)) + '\n'
            gym_split_file.write(line)
            if index % 10 == 0:
                print('%i / %i written' % (index, len(gym_data)))
        gym_split_file.close()

        return

    def extract_trajectories(self, split):
        import random

        train_valid_num_split = 1900
        if split == 'train':
            #train_valid_num_split = 200
            trajectories = [self.trajectories[int(i)] for i in self.sorted_inds[-train_valid_num_split:]]
            trajectories.extend([self.trajectories[int(i)] for i in self.sorted_inds[0:train_valid_num_split]])
        else:
            #train_valid_num_split = 1800
            trajectories = [self.trajectories[int(i)] for i in self.sorted_inds[:len(self.sorted_inds)-train_valid_num_split]]

        print('dataset %s, start to exract %i trajectories...' % (split, len(trajectories)))

        random.shuffle(trajectories)

        gym_data = []
        for i_traj, traj in enumerate(trajectories):
            len_traj = traj['rewards'].shape[0]
            if self.cfg.extract_way == 'full':
                si = min(len_traj - self.cfg.max_len, 0)
            elif self.cfg.extract_way == 'rand':
                si = random.randint(0, traj['rewards'].shape[0] - 1)
            else:
                si = 0
            if si < 0:
                continue

            print('traj=%i, len=%i, si=%i,' % (i_traj, len_traj, si))
            #s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            moving_window_step = 1
            #moving_window_step = self.cfg.max_len
            for ti in range(si, len_traj - self.cfg.max_len+1, moving_window_step):
                #print('i_traj=%i, ti=%i' % (i_traj, ti))
                s, a, r, d, rtg, timesteps, mask = self.extact_traj_window(traj, ti)
                uniq_id = '%s-traj%i-time%i' % (split, i_traj, ti)
                #example = self.process_pure_trajectory(torch.tensor(s), torch.tensor(a), torch.tensor(rtg[:,:-1,:]), uniq_id)
                gym_data.append((uniq_id, s, a, r, d, rtg, timesteps, mask))
        return gym_data

    def extact_traj_window(self, traj, si):

        # get sequences from dataset
        s = traj['observations'][si:si + self.cfg.max_len].reshape(1, -1, self.state_dim)  # shape = [1, K, s_dim]
        a = traj['actions'][si:si + self.cfg.max_len].reshape(1, -1, self.act_dim)
        r = traj['rewards'][si:si + self.cfg.max_len].reshape(1, -1, 1)
        if 'terminals' in traj:  # either terminals or dones
            d = traj['terminals'][si:si + self.cfg.max_len].reshape(1, -1)
        else:
            d = traj['dones'][si:si + self.cfg.max_len].reshape(1, -1)

        timesteps = np.arange(si, si + s.shape[1]).reshape(1, -1)  # [[si, si+1, ..., si+K-1]]
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff

        if self.cfg.scale_way == 'minmax':
            s = (s - self.state_bounds[0]) / (self.state_bounds[1] - self.state_bounds[0])
            a = (a - self.action_bounds[0]) / (self.action_bounds[1] - self.action_bounds[0])
            # r[-1] = (r[-1] - self.reward_bounds[0]) / (self.reward_bounds[1] - self.reward_bounds[0])
            # rtg[-1] = (rtg[-1] - self.return_bounds[0]) / (self.return_bounds[1] - self.return_bounds[0])

            # assert (self.state_bounds[0] <= s[-1]).all() and (self.state_bounds[1] >= s[-1]).all()
            assert (0 <= s).all() and (1 >= s).all()
            assert (0 <= a).all() and (1 >= a).all()
            # assert (0 <= r[-1]).all() and (1 >= r[-1]).all()
            # assert (0 <= rtg[-1]).all() and (1 >= rtg[-1]).all()

        rtg = discount_cumsum(traj['rewards'][si:], gamma=1.)[:s.shape[1] + 1].reshape(1, -1, 1)
        if rtg.shape[1] <= s.shape[1]:
            rtg = np.concatenate([rtg, np.zeros((1, 1, 1))], axis=1)

        if self.cfg.scale_way == 'minmax':
            #rtg_min = np.min(rtg)
            rtg_min = 0
            rtg_max = np.max(rtg)
            rtg = (rtg - rtg_min)/(rtg_max - rtg_min)

        # padding and state + reward normalization
        tlen = s.shape[1]
        s = np.concatenate([np.zeros((1, self.cfg.max_len - tlen, self.state_dim)), s], axis=1)
        if self.cfg.scale_way == 'normalize':
            s = (s - self.state_mean) / self.state_std
        a = np.concatenate([np.ones((1, self.cfg.max_len - tlen, self.act_dim)) * -10., a], axis=1)
        r = np.concatenate([np.zeros((1, self.cfg.max_len - tlen, 1)), r], axis=1)
        d = np.concatenate([np.ones((1, self.cfg.max_len - tlen)) * 2, d], axis=1)
        rtg = np.concatenate([np.zeros((1, self.cfg.max_len - tlen, 1)), rtg], axis=1) / self.scale
        timesteps = np.concatenate([np.zeros((1, self.cfg.max_len - tlen)), timesteps], axis=1)
        mask = np.concatenate([np.zeros((1, self.cfg.max_len - tlen)), np.ones((1, tlen))], axis=1)

        return s, a, r, d, rtg, timesteps, mask

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0
        if on_cloud:
            file_path = paths[(epoch - 1) % (len(paths))]
        else:
            '''
            split = 'train'
            file_path = '../../dataset/gym_data/hopper-medium-replay-v2-%s.tsv' % split
            self.convert_pickle_to_tsv(file_path, split)
            split = 'valid'
            file_path = '../../dataset/gym_data/hopper-medium-replay-v2-%s.tsv' % split
            self.convert_pickle_to_tsv(file_path, split)
            exit(3)            
            '''

            file_path = '../../dataset/gym_data/hopper-medium-replay-v2-%s.tsv' % split
            if not os.path.exists(file_path):
                self.convert_pickle_to_tsv(file_path, split)

        dataset = FileDataset(file_path, self.cfg.selected_cols, separator=self.separator)

        # GymDataset
        self.datasets[split] = GymDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            seed=self.cfg.pretrain_seed,
            code_dict_size=self.cfg.code_dict_size,
            num_bins=self.cfg.num_bins,
            patch_image_size=self.cfg.patch_image_size,
            code_image_size=self.cfg.code_image_size,
            max_image_size=self.cfg.max_image_size,
            mask_ratio=self.cfg.mask_ratio,
            random_ratio=self.cfg.random_ratio,
            keep_ratio=self.cfg.keep_ratio,
            mask_length=self.cfg.mask_length,
            poisson_lambda=self.cfg.poisson_lambda,
            replace_length=self.cfg.replace_length
        )
        return
