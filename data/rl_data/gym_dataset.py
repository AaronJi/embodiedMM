

import pickle
import math
import logging
import warnings
import numpy as np
import torch
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset
from utils.rl.rl_utils import get_nparray_from_str
from utils.vision_helper import RandomAugment
import utils.transforms as T

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
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.seed = seed
        self.code_dict_size = code_dict_size
        self.num_bins = num_bins
        self.patch_image_size = patch_image_size
        self.code_image_size = code_image_size
        self.epoch = 0
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
        self.state_dim = 11
        self.action_dim = 3
        return

    def __len__(self):
        return len(self.dataset)

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

    def load(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        self.load_traj_spec()

        return

    def process_token_seq(self, id, src, tgt):
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
            "id": id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }

        return example

    def process_trajectory_from_vars(self, id, s, a, rtg, inference=False):

        s = torch.tensor(s)
        a = torch.tensor(a)
        if not inference:
            rtg = torch.tensor(rtg[:-1])
        else:
            rtg = torch.tensor(rtg)

        r_s_a = torch.cat([rtg.reshape(-1, 1), s.reshape(-1, self.state_dim), a.reshape(-1, self.action_dim)], dim=-1).reshape(-1)

        #r_s_a_tokens = self.quantize(r_s_a, self.num_bins)
        #src_tokens = r_s_a_tokens[:-self.action_dim]
        #a_tokens = r_s_a_tokens[-self.action_dim:]

        src = r_s_a[:-self.action_dim]
        tgt = r_s_a[-self.action_dim:]
        example = self.process_token_seq(id, src, tgt)
        return example

    def process_trajectory(self, index):
        uniq_id, s, a, r, d, rtg, timesteps, mask = self.dataset[index]
        s = get_nparray_from_str(s)
        a = get_nparray_from_str(a)
        #r = get_nparray_from_str(r)
        #d = get_nparray_from_str(d)
        rtg = get_nparray_from_str(rtg)
        #timesteps = get_nparray_from_str(timesteps)
        #mask = get_nparray_from_str(mask)
        return self.process_trajectory_from_vars(uniq_id, s, a, rtg)

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
