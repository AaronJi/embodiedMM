# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import os
import random
import numpy as np
from typing import Optional
from omegaconf import DictConfig
from fairseq.tasks import register_task
from fairseq.data import Dictionary

from tasks.ofa_task import OFATask, OFAConfig
from data.rl_data.gym_dataset import GymDataset
from data.file_dataset import FileDataset
from utils.statistic_utils import cal_stat, if_array_in_bounds
from utils.rl.rl_utils import discount_cumsum, get_str_from_1darray, reform_window_len, padding_to_window

logger = logging.getLogger(__name__)
on_local = True  # if is on the local mode or cloud-computing, i.e. pai & odps

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
    gamma: float = field(
        default=0.9,
        metadata={"help": "gamma to calculate rtg from reward"},
    )
    window_len: int = field(
        default=5,
        metadata={"help": "window length of sampled trajectory data"},
    )
    moving_window_step: int = field(
        default=1,
        metadata={"help": "window moving step size"},
    )
    state_padding_num: float = field(
        default=0.0,  # 0.5
        metadata={"help": "state padding number"},
    )
    action_padding_num: float = field(
        default=-10,  # 0.5
        metadata={"help": "action padding number"},
    )
    done_padding_num: float = field(
        default=2,
        metadata={"help": "action padding number"},
    )
    extract_way: str = field(
        default='all',
        metadata={"help": "way of window generation: all, full or rand"},
    )
    scale_way: str = field(
        default='normalize',  # 'minmax'
        metadata={"help": "scale method of state and action: minmax or normalize"},
    )
    rtg_scale_num: float = field(
        default=1000.0,
        metadata={"help": "scale of return to go"},
    )
    get_stat_from_data: bool = field(
        default=True,
        metadata={"help": "if get variable statistics from data"},
    )
    only_bins: bool = field(
        default=False,
        metadata={"help": "if use only bins inside token dict"},
    )

@register_task("mujoco_control_task", dataclass=MujocoControlConfig)
class MujocoControlTask(OFATask):
    def __init__(self, cfg: MujocoControlConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.build_env()
        self.load_traj_spec()
        self.separator = "\t"

        #self.generate_data()
        return

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""
        if cfg.only_bins:
            src_dict = Dictionary()
            tgt_dict = Dictionary()
        else:
            # load dictionaries
            src_dict = cls.load_dictionary(os.path.join(cfg.bpe_dir, "dict.txt"))
            tgt_dict = cls.load_dictionary(os.path.join(cfg.bpe_dir, "dict.txt"))
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        if not cfg.only_bins:
            for i in range(cfg.code_dict_size):
                src_dict.add_symbol("<code_{}>".format(i))
                tgt_dict.add_symbol("<code_{}>".format(i))
        # quantization
        for i in range(cfg.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))
        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))
        #print(tgt_dict[58457])
        #print(tgt_dict[59456])

        return cls(cfg, src_dict, tgt_dict)

    def build_env(self):
        env = 'hopper'

        if env == 'hopper':
            self.env_name = 'Hopper-v3'
            self.max_ep_len = 1000
        elif env == 'halfcheetah':
            self.env_name = 'HalfCheetah-v3'
            self.max_ep_len = 1000
        elif env == 'walker2d':
            self.env_name = 'Walker2d-v3'
            self.max_ep_len = 1000
        elif env == 'reacher2d':
            self.env_name = 'Reacher2d'
            self.max_ep_len = 100
        else:
            raise NotImplementedError

        if on_local:
            self.build_local_env()
        else:
            self.env = None
            self.state_dim = 11
            self.action_dim = 3
        return

    def build_local_env(self):
        from environments.rl_environments.gym_environment import GymEnvironment
        self.env = GymEnvironment(None, self.env_name)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        return

    def load_traj_spec(self):
        if on_local:
            if len(self.cfg.data.split(',')) > 0:
                file_path = self.cfg.data.split(',')[0]
            else:
                file_path = self.cfg.data

            gym_file_path = '.'.join(file_path.split('.')[:-1] + ['pkl'])
            #gym_file_path = '../../dataset/gym_data/hopper-medium-replay-v2.pkl'

            import pickle
            with open(gym_file_path, 'rb') as f:
                self.trajectories = pickle.load(f)

            random.shuffle(self.trajectories)

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

        if on_local and self.cfg.get_stat_from_data:
            self.state_mean, self.state_std, self.state_bounds, self.state_norm_bounds = cal_stat(states)
            self.action_mean, self.action_std, self.action_bounds, self.action_norm_bounds = cal_stat(actions)
            self.reward_mean, self.reward_std, self.reward_bounds, self.reward_norm_bounds = cal_stat(rewards)
            self.return_mean, self.return_std, self.return_bounds, self.return_norm_bounds = cal_stat(returns)
        else:
            if on_local:
                self.state_bounds = self.env.state_bounds
                self.action_bounds = self.env.action_bounds
                #self.reward_bounds = self.env.reward_bounds
            else:
                self.state_bounds = [
                    np.array([0.7, -0.2, -1.8, -2.0, -1.0, -2.3, -5.3, -6.0, -10.0, -10.0, -10.0]),
                    np.array([1.81, 0.2, 0.06, 0.125, 1.0, 5.5, 3.4, 8.0, 9.5, 10.0, 10.0])
                ]
                self.action_bounds = [np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])]
            self.reward_bounds = [-1.5, 6.5]
            self.return_bounds = [-1.5, 3200.0]

        return

    def split_trajectories(self):
        #sample_inds = self.sorted_inds
        sample_inds = list(range(len(self.trajectories)))
        random.shuffle(sample_inds)

        train_valid_num_split = 1900
        train_valid_frac = 0.95
        train_valid_num_split = int(round(train_valid_frac*float(len(self.trajectories))))
        trajectories_train = [self.trajectories[int(i)] for i in sample_inds[:train_valid_num_split]]
        trajectories_valid = [self.trajectories[int(i)] for i in sample_inds[-(len(sample_inds)-train_valid_num_split):]]
        return trajectories_train, trajectories_valid

    def generate_data(self):
        if len(self.cfg.data.split(',')) > 0:
            file_path = self.cfg.data.split(',')[0]
        else:
            file_path = self.cfg.data

        split_data = False
        if split_data:
            trajectories_train, trajectories_valid = self.split_trajectories()

            '''
            if split == 'train':
                # train_valid_num_split = 200
                trajectories = [self.trajectories[int(i)] for i in self.sorted_inds[-train_valid_num_split:]]
                trajectories.extend([self.trajectories[int(i)] for i in self.sorted_inds[0:train_valid_num_split]])
            else:
                # train_valid_num_split = 1800
                trajectories = [self.trajectories[int(i)] for i in self.sorted_inds[:len(self.sorted_inds) - train_valid_num_split]]
    
            print('dataset %s, start to exract %i trajectories...' % (split, len(trajectories)))
            '''

            split = 'train'
            #file_path = '../../dataset/gym_data/hopper-%s-v2-%s.tsv' % (self.level, split)
            file_path = '.'.join(file_path.split('.')[:-1]) + '-' + split + '.tsv'
            print('dataset %s: %i trajectories start to write to tsv file %s...' % (split, len(trajectories_train), file_path))
            self.convert_pickle_to_tsv(file_path, trajectories_train)
            split = 'valid'
            file_path = '.'.join(file_path.split('.')[:-1]) + '-' + split + '.tsv'
            print('dataset %s: %i trajectories start to write to tsv file %s...' % (split, len(trajectories_valid), file_path))
            self.convert_pickle_to_tsv(file_path, trajectories_valid)
        else:
            print('%i trajectories start to write to tsv file %s...' % (len(self.trajectories), file_path))
            self.convert_pickle_to_tsv(file_path, self.trajectories)

        exit(3)
        return

    def convert_pickle_to_tsv(self, file_path, trajectories):
        gym_data = self.extract_trajectories(trajectories, file_path)

        gym_data_file = open(file_path, 'w')
        for index in range(len(gym_data)):
            uniq_id, s, a, r, d, rtg, timesteps, mask = gym_data[index]
            line = uniq_id + self.separator + get_str_from_1darray(s.reshape(-1)) + self.separator + get_str_from_1darray(a.reshape(-1)) + self.separator + get_str_from_1darray(
                r.reshape(-1)) + self.separator + get_str_from_1darray(d.reshape(-1)) + self.separator + get_str_from_1darray(rtg.reshape(-1)) + self.separator + get_str_from_1darray(
                timesteps.reshape(-1)) + self.separator + get_str_from_1darray(mask.reshape(-1)) + '\n'
            gym_data_file.write(line)
            if index % 10 == 0:
                print('%i / %i written' % (index, len(gym_data)))
        gym_data_file.close()

        return

    def extract_trajectories(self, trajectories, file_path):
        gym_data = []
        for i_traj, traj in enumerate(trajectories):
            len_traj = len(traj['rewards'])
            print('traj=%i, len=%i' % (i_traj, len_traj))
            for t_end_window in range(len_traj):
                s, a, r, d, rtg, timesteps, mask = self.extact_traj_window(traj, t_end_window, self.cfg.window_len)

                if self.cfg.scale_way == 'normalize':
                    s = (s - self.state_mean) / self.state_std
                    a = (a - self.action_mean) / self.action_std
                else:
                    s = (s - self.state_bounds[0]) / (self.state_bounds[1] - self.state_bounds[0])
                    a = (a - self.action_bounds[0]) / (self.action_bounds[1] - self.action_bounds[0])
                rtg = rtg/self.cfg.rtg_scale_num

                uniq_id = '%s-traj%i-endtime%i' % (file_path.split('.')[-2], i_traj, t_end_window)
                #print(uniq_id, s, a, r, d, rtg, timesteps, mask)
                #exit(5)
                gym_data.append((uniq_id, s, a, r, d, rtg, timesteps, mask))

                #if i_traj > 100:
                #    break
        return gym_data

    def extact_traj_window(self, traj, t_end_window, window_len):
        # get sequences from dataset
        s = reform_window_len(traj['observations'], window_len, index_end=t_end_window, padding_num=self.cfg.state_padding_num)
        a = reform_window_len(traj['actions'], window_len, index_end=t_end_window, padding_num=self.cfg.action_padding_num)
        r = reform_window_len(traj['rewards'], window_len, index_end=t_end_window, padding_num=0.0, batch_dim=0)
        d = reform_window_len(traj['terminals' if 'terminals' in traj else 'dones'], window_len, index_end=t_end_window, padding_num=self.cfg.done_padding_num, batch_dim=0)
        assert len(s) > 0 and len(a) > 0 and len(r) > 0 and len(d) > 0

        t_start_window = max(t_end_window - window_len + 1, 0)
        t_len = t_end_window - t_start_window + 1

        timesteps = np.arange(t_start_window, t_end_window + 1)  # [[si, si+1, ..., si+K-1]]
        timesteps = padding_to_window(timesteps, window_len, padding_num=0, batch_dim=0)
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff

        # padding and state + reward normalization
        #rtg_padding_num = float(rtg[:, 0, 0])
        rtg_padding_num = 0

        returns = discount_cumsum(traj['rewards'][t_start_window:], gamma=self.cfg.gamma)
        #rtg = returns[:t_len + 1]
        #if rtg.shape[0] <= t_len:
        #    rtg = np.concatenate([rtg, np.zeros(1)], axis=0)  # TODO why?
        rtg = returns[:t_len]

        rtg = padding_to_window(rtg, window_len, padding_num=rtg_padding_num, batch_dim=0)

        mask = np.concatenate([np.ones(self.cfg.window_len - t_len), np.zeros(t_len)], axis=0)  # TODO different definition with DT paper!

        return s, a, r, d, rtg, timesteps, mask


    def extract_trajectories0(self, trajectories, file_path):
        gym_data = []
        for i_traj, traj in enumerate(trajectories):
            len_traj = len(traj['rewards'])
            if self.cfg.extract_way == 'full':
                t0_start = min(len_traj - self.cfg.window_len, 0)
            elif self.cfg.extract_way == 'rand':
                import random
                t0_start = random.randint(0, traj['rewards'].shape[0] - 1)
            else:
                #t0_start = 1 - self.cfg.window_len
                t0_start = 0
            if t0_start < 1 - self.cfg.window_len:
                continue

            t0_end = max(len_traj - self.cfg.window_len+1, t0_start)
            print('traj=%i, len=%i, t0_start=%i, t0_end=%i' % (i_traj, len_traj, t0_start, t0_end))
            moving_window_step = 1
            #moving_window_step = self.cfg.window_len
            for t0 in range(t0_start, t0_end, moving_window_step):
                #print('i_traj=%i, ti=%i' % (i_traj, ti))
                s, a, r, d, rtg, timesteps, mask = self.extact_traj_window0(traj, t0)
                if self.cfg.scale_way == 'minmax':
                    if not if_array_in_bounds(s, [0.0, 1.0]):
                        print('state out of bound, skip')
                        continue
                    if not if_array_in_bounds(a, [0.0, 1.0]):
                        print('action out of bound, skip')
                        continue
                    if not if_array_in_bounds(rtg, [0.0, 1.0]):
                        print('rtg out of bound, skip')
                        print(rtg)
                        continue

                uniq_id = '%s-traj%i-time%i' % (file_path.split('.')[-2], i_traj, t0)
                gym_data.append((uniq_id, s, a, r, d, rtg, timesteps, mask))

        return gym_data

    def extact_traj_window0(self, traj, si):
        # get sequences from dataset
        s = traj['observations'][si:si + self.cfg.window_len].reshape(1, -1, self.state_dim)  # shape = [1, K, s_dim]
        a = traj['actions'][si:si + self.cfg.window_len].reshape(1, -1, self.action_dim)
        r = traj['rewards'][si:si + self.cfg.window_len].reshape(1, -1, 1)
        if 'terminals' in traj:  # either terminals or dones
            d = traj['terminals'][si:si + self.cfg.window_len].reshape(1, -1)
        else:
            d = traj['dones'][si:si + self.cfg.window_len].reshape(1, -1)
        assert len(s) > 0 and len(a) > 0 and len(r) > 0 and len(d) > 0

        timesteps = np.arange(max(si, 0), max(si, 0) + s.shape[1]).reshape(1, -1)  # [[si, si+1, ..., si+K-1]]
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff

        if self.cfg.scale_way == 'minmax':
            s = (s - self.state_bounds[0]) / (self.state_bounds[1] - self.state_bounds[0])
            a = (a - self.action_bounds[0]) / (self.action_bounds[1] - self.action_bounds[0])

            # assert (self.state_bounds[0] <= s[-1]).all() and (self.state_bounds[1] >= s[-1]).all()
            # assert (0 <= s).all() and (1 >= s).all()
            # assert (0 <= a).all() and (1 >= a).all()

        rtg = discount_cumsum(traj['rewards'][si:], gamma=self.cfg.gamma)[:s.shape[1] + 1].reshape(1, -1, 1)
        if rtg.shape[1] <= s.shape[1]:
            rtg = np.concatenate([rtg, np.zeros((1, 1, 1))], axis=1)

        if self.cfg.scale_way == 'minmax':
            rtg_min = min(0.0, np.min(rtg))
            #rtg_min = np.min(rtg)
            #rtg_max = np.min(discount_cumsum(traj['rewards'], gamma=self.cfg.gamma))
            #rtg_max = np.max(rtg)
            rtg_max = np.max(discount_cumsum(traj['rewards'], gamma=self.cfg.gamma))
            rtg = (rtg - rtg_min)/(rtg_max - rtg_min)
        else:
            rtg_min = min(0.0, np.min(rtg))
            rtg_max = np.max(discount_cumsum(traj['rewards'], gamma=self.cfg.gamma))
            rtg = (rtg - rtg_min) / (rtg_max - rtg_min)

        # padding and state + reward normalization
        #rtg_padding_num = float(rtg[:, 0, 0])
        rtg_padding_num = 0
        tlen = s.shape[1]
        s = np.concatenate([self.cfg.state_padding_num*np.ones((1, self.cfg.window_len - tlen, self.state_dim)), s], axis=1)
        if self.cfg.scale_way == 'normalize':
            s = (s - self.state_mean) / self.state_std
        a = np.concatenate([self.cfg.action_padding_num*np.ones((1, self.cfg.window_len - tlen, self.action_dim)), a], axis=1)
        if self.cfg.scale_way == 'normalize':
            a = (a - self.action_mean) / self.action_std
        r = np.concatenate([np.zeros((1, self.cfg.window_len - tlen, 1)), r], axis=1)
        d = np.concatenate([np.ones((1, self.cfg.window_len - tlen)) * 2, d], axis=1)
        rtg = np.concatenate([rtg_padding_num*np.ones((1, self.cfg.window_len - tlen, 1)), rtg], axis=1) # / self.scale
        timesteps = np.concatenate([np.zeros((1, self.cfg.window_len - tlen)), timesteps], axis=1)
        mask = np.concatenate([np.ones((1, self.cfg.window_len - tlen)), np.zeros((1, tlen))], axis=1)  # TODO different definition with DT paper!

        return s, a, r, d, rtg, timesteps, mask

    def load_local_dataset(self, split, file_path):

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
            replace_length=self.cfg.replace_length,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            state_padding_num=self.cfg.state_padding_num,
            action_padding_num=self.cfg.action_padding_num
        )
        return

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]

        if on_local:
            #if not os.path.exists(file_path):
            #    self.generate_data()
            dataset = FileDataset(file_path, self.cfg.selected_cols, separator=self.separator)
        else:
            from data.odps_dataset import TableDataset
            dataset = TableDataset(file_path, self.cfg.selected_cols)

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
            replace_length=self.cfg.replace_length,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            state_padding_num=self.cfg.state_padding_num,
            action_padding_num=self.cfg.action_padding_num
        )
        #self.check_data(self.datasets[split])
        #self.check_data_shape(self.datasets[split])
        return

    def print_model_params(self, model):
        import torch
        print('&'*50)
        for name, parameters in model.named_parameters():
            #print(name, ':', parameters.size())
            print(name, ':', torch.norm(parameters))
        print('&' * 50)
        return

    def check_data(self, check_dataset):
        for index in range(len(check_dataset)):
            sample = check_dataset[index]
            #print(index, sample['id'])
            index_base = 3

            from data.rl_data.gym_dataset import print_sample
            print_sample(sample)

            print(sample['id'].split('-'))

            traj_num = int(sample['id'].split('-')[index_base][len('traj'):])
            time_num = int(sample['id'].split('-')[index_base+1][len('endtime'):])
            print(traj_num, time_num)

            exit(4)

            #if (traj_num == 453 and time_num > 140) or traj_num >= 454:
            check_traj = 453
            check_traj = 134
            #if traj_num >= 453:
            #if traj_num == check_traj:
            #    print_sample(sample)
            #if traj_num > check_traj:
            #    exit(5)
        exit(4)
        # '/dataset/gym_data/hopper-medium-replay-v2-traj453-time140']

        return
    def check_data_shape(self, check_dataset):
        print(len(check_dataset))
        #print(check_dataset['0'])

        for index in range(len(check_dataset)):
            sample = check_dataset[index]

            '''
            print(sample['id'])
            print(sample['source'])
            print(sample['source_mask'])
            print(sample['prev_output_tokens'])
            print(sample['target'])
            print(sample['target_mask'])
            print(sample['source_time'])
            exit(3)
            '''

            print(index, sample['id'])
            print('source shape is %s, target shape is %s, prev target shape is %s' % (sample['source'].shape == (20, 15), sample['target'].shape == (20, 3), sample['prev_output_tokens'].shape == (20, 3)))
            print('source mask shape is %s, target mask shape is %s, details: %s, %s' % (sample['source_mask'].shape[0]== 20, sample['target_mask'].shape[0] == 20, sample['source_mask'].shape, sample['target_mask'].shape))
            print('masks: %s, %s' % ((1-sample['source_mask']).any(), (1 - sample['target_mask']).any()))
            #print((1-sample['source_mask']).any())
            #print((1 - sample['target_mask']).any())

            if sample['source'].shape != (20, 15) or sample['target'].shape != (20, 3) or sample['prev_output_tokens'].shape != (20, 3) or sample['source_mask'].shape[0] != 20 or sample['target_mask'].shape[0] == 20 != 20:
                print('shape error!')
                print(sample['source'].shape != (20, 15))
                print(sample['target'].shape != (20, 3))
                print(sample['prev_output_tokens'].shape != (20, 3))
                print(sample['source_mask'].shape[0] != 20)
                print(sample['target_mask'].shape[0] != 20)


                print(sample['id'])
                print(sample['source'].shape)
                print(sample['source_mask'].shape)
                print(sample['target'].shape)
                print(sample['target_mask'].shape)
                print(sample['prev_output_tokens'].shape)

                exit(2)
            if not ((1-sample['source_mask']).any() and (1 - sample['target_mask']).any()):
                print('mask error!')
                print(sample['id'])
                print(sample['source_mask'])
                print(sample['target_mask'])

                exit(2)
        exit(2)
        return

    '''
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, **extra_kwargs
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        import torch
        from fairseq.optim.amp_optimizer import AMPOptimizer
        #print(sample['id'])
        #print(sample['net_input']['sources'])
        #print(sample['target'])
        #print('&&&&& source_eff=%f, target_eff=%f' % (torch.sum(1 - sample['net_input']['source_masks']).data, torch.sum(1 - sample['target_mask']).data))
        #print('update_num=%i' % update_num)
        #self.print_model_params(model)
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, update_num=update_num)

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output    
    '''

