
import argparse
import pickle
import random
import numpy as np
import torch

from data.ofa_dataset import OFADataset
from utils.rl.rl_utils import discount_cumsum
from environments.rl_environments.gym_environment import GymEnvironment
from utils.statistic_utils import cal_stat

class GymDataset(OFADataset):
    def __init__(self, cfg: argparse.Namespace, env: GymEnvironment, max_ep_len=1000, scale=1):
        self.cfg = cfg
        self.env = env
        self.trajectories = None
        self.device = cfg.device
        self.max_ep_len = max_ep_len
        self.scale = scale

        return


    def load(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        self.load_traj_spec()

        return

    def get_batch(self, batch_size=256, max_len=20, scale_way='normalize'):
        batch_inds = np.random.choice(
            np.arange(self.num_trajectories),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = self.trajectories[int(self.sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, self.env.state_dim))   # shape = [1, K, s_dim]
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, self.env.action_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:  # either terminals or dones
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))  # [[si, si+1, ..., si+K-1]]
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len-1  # padding cutoff

            if scale_way == 'standardize':
                s[-1] = (s[-1] - self.state_bounds[0]) / (self.state_bounds[1] - self.state_bounds[0])
                a[-1] = (a[-1] - self.action_bounds[0]) / (self.action_bounds[1] - self.action_bounds[0])
                #r[-1] = (r[-1] - self.reward_bounds[0]) / (self.reward_bounds[1] - self.reward_bounds[0])
                #rtg[-1] = (rtg[-1] - self.return_bounds[0]) / (self.return_bounds[1] - self.return_bounds[0])

                #assert (self.state_bounds[0] <= s[-1]).all() and (self.state_bounds[1] >= s[-1]).all()
                assert (0 <= s[-1]).all() and (1 >= s[-1]).all()
                assert (0 <= a[-1]).all() and (1 >= a[-1]).all()
                #assert (0 <= r[-1]).all() and (1 >= r[-1]).all()
                #assert (0 <= rtg[-1]).all() and (1 >= rtg[-1]).all()

            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))

            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, self.env.state_dim)), s[-1]], axis=1)
            if scale_way == 'normalize':
                s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, self.env.action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)
        return s, a, r, d, rtg, timesteps, mask

    def load_traj_spec(self):
        self.mode = 'normal'
        self.pct_traj = 1.0
        self.max_len = 20

        states, actions, rewards, returns, traj_lens = [], [], [], [], []
        for path in self.trajectories:
            if self.mode == 'delayed':  # delayed: all rewards moved to end of trajectory
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

        pct_traj = self.pct_traj

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