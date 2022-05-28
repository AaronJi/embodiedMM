

from io import BytesIO


import argparse
import pickle
import math
import logging
import random
import warnings

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset
from utils.rl.rl_utils import discount_cumsum
from utils.statistic_utils import cal_stat
<<<<<<< Updated upstream
=======
from utils.vision_helper import RandomAugment
import utils.transforms as T

from environments.rl_environments.gym_environment import GymEnvironment


ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_whole_word_mask(bpe, dictionary):
    if bpe is not None:

        def is_beginning_of_word(i):
            if i < dictionary.nspecial:
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return bpe.is_beginning_of_word(tok)
            except ValueError:
                return True
>>>>>>> Stashed changes

        mask_whole_words = torch.ByteTensor(
            list(map(is_beginning_of_word, range(len(dictionary))))
        )
        return mask_whole_words
    return None


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
    }

    return batch

class GymDataset(OFADataset):
    def __init__(self, cfg: argparse.Namespace, env: GymEnvironment, max_ep_len=1000, scale=1):
        self.cfg = cfg
        self.env = env
        self.trajectories = None
        self.device = cfg.device
        self.max_ep_len = max_ep_len
        self.scale = scale

        return

    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        seed=7,
        code_dict_size=8192,
        num_bins=1000,
        patch_image_size=384,
        code_image_size=128,
        pure_text_dataset=None,
        pure_image_dataset=None,
        detection_dataset=None,
        all_object_list=None,
        all_caption_list=None,
        type2ans_dict=None,
        ans2type_dict=None,
        max_image_size=512,
        mask_ratio=0.3,
        random_ratio=0.0,
        keep_ratio=0.0,
        mask_length="span-poisson",
        poisson_lambda=3.0,
        replace_length=1
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.seed = seed
        self.code_dict_size = code_dict_size
        self.num_bins = num_bins
        self.patch_image_size = patch_image_size
        self.code_image_size = code_image_size

        self.pure_text_dataset = pure_text_dataset
        self.pure_image_dataset = pure_image_dataset
        self.detection_dataset = detection_dataset
        self.epoch = 0

        self.all_object_list = all_object_list
        self.all_caption_list = all_caption_list
        self.type2ans_dict = type2ans_dict
        self.ans2type_dict = ans2type_dict

        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.keep_ratio = keep_ratio
        self.mask_length = mask_length
        self.poisson_lambda = poisson_lambda
        self.replace_length = replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if self.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={self.mask_length}")
        if self.mask_length == "subword" and self.replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace-length=1 or 0")

        self.mask_idx = src_dict.index("<mask>")
        self.mask_whole_word = (
            get_whole_word_mask(self.bpe, self.src_dict)
            if self.mask_length != "subword"
            else None
        )
        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            _lambda = self.poisson_lambda
            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)


        self.pos_tgt_item = self.encode_text(" yes")
        self.neg_tgt_item = self.encode_text(" no")

        self.mask_left = self.mask_top = int(0.5 * self.code_image_size)
        self.mask_right = self.mask_bottom = int(1.5 * self.code_image_size)
        self.mask_ids = [
            i*self.code_image_size*2+j
            for i in range(self.code_image_size*2) for j in range(self.code_image_size*2)
            if not (self.mask_left <= i < self.mask_right and self.mask_top <= j < self.mask_bottom)
        ]

        scales = np.arange(patch_image_size, 481).tolist()

        # for image-text pair
        self.patch_resize_transform = transforms.Compose([
            T.RandomResize(scales, max_size=672),
            transforms.CenterCrop(patch_image_size),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # for pure image
        self.patch_crop_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # for detection
        self.detection_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.LargeScaleJitter(output_size=self.code_image_size*2, aug_scale_min=1.0, aug_scale_max=1.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=max_image_size)
        ])
        # for visual grounding
        self.visual_grounding_transform = T.Compose([
            T.RandomResize(scales, max_size=672),
            T.ObjectCenterCrop((patch_image_size, patch_image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=max_image_size)
        ])

        self.device = 'cpu'

        self.build_env()
        self.load_traj_spec()
        self.extract_traj()

        #print(len(self.dataset))
        #print(len(self.gym_dataset))

        #print('*'*10)
        #exit(4)
        return

    def __len__(self):
        return len(self.gym_dataset)

    def __getitem__(self, index):

        #with data_utils.numpy_seed(self.seed, self.epoch):
        example = self.process_trajectory(index)

        return example


    def collater(self, samples, pad_to_length=None):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        return collate(samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos)

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

<<<<<<< Updated upstream
=======
        if self.model == 'bc':
            self.env_targets = self.env_targets[:1]  # since BC ignores target, no need for different evaluations

        self.env = GymEnvironment(None, self.env_name)

        return


    def load_traj_spec(self):
        self.mode = 'normal'
        self.pct_traj = 1.0
        self.max_len = 20

        self.trajectories = self.dataset
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream

        '''
        print(states.shape)
        print(self.state_mean)
        print(self.state_std)
        print(self.state_bounds)
        print(self.num_timesteps)
        print(self.reward_bounds)
        print(self.return_bounds)
        exit(4)
        '''
=======
>>>>>>> Stashed changes

        '''
        print(states.shape)
        print(self.state_mean)
        print(self.state_std)
        print(self.state_bounds)
        print(self.num_timesteps)
        print(self.reward_bounds)
        print(self.return_bounds)
        exit(4)
        '''

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

<<<<<<< Updated upstream
        return

=======
    def load(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        self.load_traj_spec()

        return

    def extract_traj(self):
        extract_way = 'rand'
        extract_way = 'full'
        extract_way = 'all'

        self.gym_dataset = []

        print('dataset %s, start to exract %i trajectories...' % (self.split, len(self.dataset)))
        for i_traj, traj in enumerate(self.dataset):

            len_traj = traj['rewards'].shape[0]

            if extract_way == 'full':
                si = min(len_traj - self.max_len, 0)
            elif extract_way == 'rand':
                si = random.randint(0, traj['rewards'].shape[0] - 1)
            else:
                si = 0

            if si < 0:
                continue

            #len_traj = 21
            #print(len_traj)
            print('traj=%i, len=%i, si=%i,' % (i_traj, len_traj, si))
            #s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            for ti in range(si, len_traj - self.max_len+1):
                #print('i_traj=%i, ti=%i' % (i_traj, ti))
                s, a, r, d, rtg, timesteps, mask = self.extact_traj_window(traj, ti)
                uniq_id = '%s-traj%i-time%i' % (self.split, i_traj, ti)
                #example = self.process_pure_trajectory(torch.tensor(s), torch.tensor(a), torch.tensor(rtg[:,:-1,:]), uniq_id)
                self.gym_dataset.append([uniq_id, s, a, r, d, rtg, timesteps, mask])
        return

    def extact_traj_window(self, traj, si):
        scale_way = 'normalize'
        scale_way = 'minmax'

        # get sequences from dataset
        s = traj['observations'][si:si + self.max_len].reshape(1, -1, self.env.state_dim)  # shape = [1, K, s_dim]
        a = traj['actions'][si:si + self.max_len].reshape(1, -1, self.env.act_dim)
        r = traj['rewards'][si:si + self.max_len].reshape(1, -1, 1)
        if 'terminals' in traj:  # either terminals or dones
            d = traj['terminals'][si:si + self.max_len].reshape(1, -1)
        else:
            d = traj['dones'][si:si + self.max_len].reshape(1, -1)

        timesteps = np.arange(si, si + s.shape[1]).reshape(1, -1)  # [[si, si+1, ..., si+K-1]]
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff

        if scale_way == 'minmax':
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

        if scale_way == 'minmax':
            rtg_min = np.min(rtg)
            rtg_max = np.max(rtg)
            rtg = (rtg - rtg_min)/(rtg_max - rtg_min)

        # padding and state + reward normalization
        tlen = s.shape[1]
        s = np.concatenate([np.zeros((1, self.max_len - tlen, self.env.state_dim)), s], axis=1)
        if scale_way == 'normalize':
            s = (s - self.state_mean) / self.state_std
        a = np.concatenate([np.ones((1, self.max_len - tlen, self.env.act_dim)) * -10., a], axis=1)
        r = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r], axis=1)
        d = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d], axis=1)
        rtg = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg], axis=1) / self.scale
        timesteps = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps], axis=1)
        mask = np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1)

        return s, a, r, d, rtg, timesteps, mask


    def process_trajectory(self, index):
        uniq_id = self.gym_dataset[index][0]
        s = self.gym_dataset[index][1]
        a = self.gym_dataset[index][2]
        #r = self.gym_dataset[index][3]
        #d = self.gym_dataset[index][4]
        rtg = self.gym_dataset[index][5]
        #timesteps = self.gym_dataset[index][6]
        #mask = self.gym_dataset[index][7]

        patch_image = torch.zeros((3, self.code_image_size*2, self.code_image_size*2))
        patch_mask = torch.tensor([False])
        code_mask = torch.tensor([False])
        conf = torch.tensor([1.0])

        s = torch.tensor(s)
        a = torch.tensor(a)
        rtg = torch.tensor(rtg[:, :-1, :])

        r_s = torch.cat([rtg, s], dim=2)
        r_s_tokens = self.quantize(r_s.reshape(-1), self.num_bins)
        a_tokens = self.quantize(a.reshape(-1), self.num_bins)

        src_item = torch.cat([self.bos_item, r_s_tokens, self.eos_item])
        target_item = torch.cat([a_tokens, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, a_tokens])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }
        return example

    def quantize(self, tensor_v_rel, num_bins):
        q_tokens = ["<bin_{}>".format(int((v_rel * (num_bins - 1)).round())) for v_rel in tensor_v_rel]
        q_item = self.encode_text(' '.join(q_tokens), use_bpe=False)
        return q_item

>>>>>>> Stashed changes
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
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, self.env.act_dim))
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
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, self.env.act_dim)) * -10., a[-1]], axis=1)
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
