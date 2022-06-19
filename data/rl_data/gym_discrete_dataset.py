

import pickle
import math
import logging
import warnings
import numpy as np
import torch
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.rl_data.gym_dataset import GymDataset
from utils.rl.rl_utils import get_nparray_from_str
from utils.vision_helper import RandomAugment
import utils.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

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

class GymDiscreteDataset(GymDataset):
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
        max_image_size=512,
        mask_ratio=0.3,
        random_ratio=0.0,
        keep_ratio=0.0,
        mask_length="span-poisson",
        poisson_lambda=3.0,
        replace_length=1
    ):
        super().__init__(split,
        dataset,
        bpe,
        src_dict,
        tgt_dict,
        max_src_length,
        max_tgt_length,
        seed,
        code_dict_size,
        num_bins,
        patch_image_size,
        code_image_size,
        max_image_size,
        mask_ratio,
        random_ratio,
        keep_ratio,
        mask_length,
        poisson_lambda,
        replace_length)
        return

    def __getitem__(self, index):
        #with data_utils.numpy_seed(self.seed, self.epoch):
        uniq_id, src, tgt, timestep, mask = self.process_trajectory(index)
        example = self.process_token_seq(uniq_id, src, tgt)
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        return collate(samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos)

    def load(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        self.load_traj_spec()

        return

    def process_trajectory_from_vars(self, uniq_id, s, a, r, d, rtg, timesteps, mask, inference=False):

        s = torch.tensor(s)
        a = torch.tensor(a)
        if not inference:
            rtg = torch.tensor(rtg[:-1])
        else:
            rtg = torch.tensor(rtg)

        rsa = torch.cat([rtg.reshape(-1, 1), s.reshape(-1, self.state_dim), a.reshape(-1, self.action_dim)], dim=-1).reshape(-1)

        src = rsa[:-self.action_dim]
        tgt = rsa[-self.action_dim:]

        pos = timesteps.reshape((-1, 1))
        return uniq_id, src, tgt, pos, mask.reshape((-1, 1))

    def process_token_seq(self, uniq_id, src, tgt):
        src_tokens = self.quantize(src, self.num_bins)
        use_end_token = False
        if use_end_token:
            src_item = torch.cat([self.bos_item, src_tokens, self.eos_item])
        else:
            src_item = src_tokens

        if tgt is None:
            target_item = self.eos_item.reshape(-1)
            prev_output_item = self.bos_item.reshape(-1)
        else:
            tgt_tokens = self.quantize(tgt, self.num_bins)
            target_item = torch.cat([tgt_tokens, self.eos_item])
            prev_output_item = torch.cat([self.bos_item, tgt_tokens])

        patch_image = torch.zeros((3, self.code_image_size * 2, self.code_image_size * 2))
        patch_mask = torch.tensor([False])
        code_mask = torch.tensor([False])
        conf = torch.tensor([1.0])

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
        q_tokens = []
        for v_rel in tensor_v_rel:
            try:
                iv = int((v_rel * (num_bins - 1)).round())
            except:
                iv = 999
            q_tokens.append("<bin_{}>".format(iv))
        #q_tokens = ["<bin_{}>".format(iv) for v_rel in tensor_v_rel]
        q_item = self.encode_text(' '.join(q_tokens), use_bpe=False)
        return q_item
