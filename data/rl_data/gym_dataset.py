

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


def print_sample(sample):
    print('id: ', sample['id'])
    print('source: ', sample['source'])
    print('source_mask: ', sample['source_mask'])
    print('prev_output_tokens: ', sample['prev_output_tokens'])
    print('target: ', sample['target'])
    print('target_mask: ', sample['target_mask'])
    print('source_time: ', sample['source_time'].reshape(-1))
    return


def print_batch(samples):
    print('id: ', samples['id'])
    print('nsentences: ', samples['nsentences'])
    print('ntokens: ', samples['ntokens'])
    print('src_tokens: ', samples['net_input']['src_tokens'])
    print('src_lengths: ', samples['net_input']['src_lengths'])
    print('sources: ', samples['net_input']['sources'])
    print('source_lengths: ', samples['net_input']['source_lengths'])
    print('source_masks: ', samples['net_input']['source_masks'])
    print('source_times: ', samples['net_input']['source_times'])
    print('prev_output_tokens: ', samples['net_input']['prev_output_tokens'])
    print('target: ', samples['target'])
    print('target_mask: ', samples['target_mask'])
    print('target_length: ', samples['target_length'].reshape(-1))
    return

def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}
    '''
    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )
    '''

    def merge(key):
        return torch.stack([s[key] for s in samples], dim=0)

    id = np.array([s["id"] for s in samples])
    #src_tokens = merge("source")
    #src_tokens = torch.stack([s["source"] for s in samples], dim=0)
    dummy_src_token = torch.tensor([], dtype=torch.int)
    src_tokens = torch.stack([dummy_src_token for s in samples], dim=0)  # torch.zeros((4, 0))
    src_lengths = torch.LongTensor([dummy_src_token.ne(pad_idx).long().sum() for s in samples])

    sources = merge("source")
    source_lengths = torch.LongTensor([s["source_mask"].ne(1).long().sum() for s in samples])
    source_masks = merge("source_mask")
    source_times = merge("source_time")

    nsentences = len(samples)
    #ntokens = (code_masks != 0).sum().detach().cpu().item()

    #patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    #patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    target_mask = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        target_mask = merge("target_mask")
        #tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        tgt_lengths = torch.LongTensor([s["target_mask"].ne(1).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()
        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()
        #ntokens = (code_masks != 0).sum().detach().cpu().item()
        tgt_lengths = None

    batch = {
        "id": id,
        "nsentences": nsentences,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "sources": sources,
            "source_lengths": source_lengths,
            "source_masks": source_masks,
            "source_times": source_times,
            #"patch_images": patch_images,
            #"patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
        "target_length": tgt_lengths,
        "target_mask": target_mask,
    }

    '''
    print(batch["id"])
    print(batch["nsentences"])
    print(batch["ntokens"])
    print(batch["net_input"]["obs"])
    print(batch["net_input"]["obs_lengths"])
    print(batch["net_input"]["code_times"])
    print(batch["net_input"]["code_masks"])
    print(batch["net_input"]["prev_output_tokens"])
    print(batch["target"])
    print(batch["target_length"])
    print(batch["target_mask"])
    print('&'*10)
    print(batch["net_input"]["obs"].shape)
    print(batch["net_input"]["obs_lengths"].shape)
    print(batch["net_input"]["code_times"].shape)
    print(batch["net_input"]["code_masks"].shape)
    print(batch["net_input"]["prev_output_tokens"].shape)
    print(batch["target"].shape)
    print(batch["target_length"].shape)
    print(batch["target_mask"].shape)
    exit(4) 
    '''

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
        replace_length=1,
        state_dim=11,
        action_dim=3,
        state_padding_num=0.0,
        action_padding_num=0.0
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
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_padding_num = state_padding_num
        self.action_padding_num = action_padding_num

        '''
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
        
        '''

        #self.device = 'cpu'
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        #with data_utils.numpy_seed(self.seed, self.epoch):
        uniq_id, src, tgt, timestep, mask = self.process_trajectory(index)
        example = self.process_train_seq(uniq_id, src, tgt, timestep, mask)
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

    def process_train_seq(self, uniq_id, src, tgt, timestep, mask):
        prev_tgt = torch.cat([self.action_padding_num*torch.ones(1, self.action_dim), tgt[:-1]])
        src_mask = mask
        tgt_mask = mask

        example = {
            "id": uniq_id,
            "source": src,
            "source_mask": src_mask,
            "source_time": timestep,
            #"patch_image": patch_image,
            #"patch_mask": patch_mask,

            "target": tgt,
            "target_mask": tgt_mask,
            "prev_output_tokens": prev_tgt,
            #"conf": conf,
        }

        return example

    def process_trajectory_from_vars(self, uniq_id, s, a, r, d, rtg, timesteps, mask, inference=False):

        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        if not inference:
            #rtg = torch.tensor(rtg[:-1], dtype=torch.float32)
            rtg = torch.tensor(rtg, dtype=torch.float32)
        else:
            rtg = torch.tensor(rtg, dtype=torch.float32)
        timesteps = torch.tensor(timesteps).reshape((-1, 1))

        #rsa = torch.cat([rtg.reshape(-1, 1), s.reshape(-1, self.state_dim), a.reshape(-1, self.action_dim)], dim=-1).reshape(-1)
        #src = rsa[:-self.action_dim]
        #tgt = rsa[-self.action_dim:]


        mask = torch.tensor(mask)
        #mask = 1 - mask  # TODO
        #tgt_mask = mask*torch.ones((1, self.action_dim))

        tgt = a.reshape((-1, self.action_dim))  # shape = [T, dim_a]
        train_a = tgt.clone()
        train_a[-1] = self.action_padding_num*torch.ones(self.action_dim)

        rsa = torch.cat([rtg.reshape(-1, 1), s.reshape(-1, self.state_dim), train_a], dim=-1)  # shape = [T, 1 + dim_s + dim_a]
        #src = rsa[:-self.action_dim].reshape(-1, 1)  # shape = [1, T*(1+dim_s) + (T-1)*dim_s]
        src = rsa  # shape = [T, 1 + dim_s + dim_a]
        #src_mask = torch.cat([mask*torch.ones((1, 1)), mask*torch.ones((1, self.state_dim)), mask*torch.ones((1, self.action_dim))], dim=-1)

        return uniq_id, src, tgt, timesteps, mask

    def process_trajectory(self, index):
        uniq_id, s, a, r, d, rtg, timesteps, mask = self.dataset[index]
        s = get_nparray_from_str(s)
        a = get_nparray_from_str(a)
        r = get_nparray_from_str(r)
        d = get_nparray_from_str(d)
        rtg = get_nparray_from_str(rtg)
        timesteps = get_nparray_from_str(timesteps)
        mask = get_nparray_from_str(mask)

        uniq_id, src, tgt, timesteps, mask = self.process_trajectory_from_vars(uniq_id, s, a, r, d, rtg, timesteps, mask)
        return uniq_id, src, tgt, timesteps, mask

if __name__ == '__main__':
    from data.file_dataset import FileDataset

    env = 'hopper'
    #dataset = 'expert'
    dataset = 'medium-replay'
    data_dir = "/Users/jiluo-wendu/git/myGit/embodiedMM/dataset/gym_data"
    file_path = "%s/%s-%s-v2.tsv" % (data_dir, env, dataset)
    selected_cols = '0,1,2,3,4,5,6,7'
    separator = '"\t"'

    print(file_path)
    dataset = FileDataset(file_path, selected_cols, separator=separator)

    bpe = {}

    from fairseq.data import Dictionary
    src_dict = Dictionary()
    tgt_dict = Dictionary()
    gym_dataset = GymDataset(
        'test',
        dataset,
        bpe,
        src_dict
    )

    print(len(gym_dataset))
    for index in range(len(gym_dataset)):

        sample = gym_dataset[index]
        print(sample)






