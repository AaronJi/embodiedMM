# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from dataclasses import field
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rectify_rot6d(x):
    """
    Input:
        (B,6) Batch of *approximate* 6-D rotation representations
    Output:
        (B,6) Batch of *valid* 6-D rotation representations
    """
    a = x.reshape(-1, 3, 2)
    a1 = a[:, :, 0]
    a2 = a[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b = torch.stack((b1, b2), dim=-1)
    y = b.reshape(*x.shape)
    return y


def rectify_poses(trans_and_poses):
    """
    Input:
        (B,T,3+J*6)
    Output:
        (B,T,3+J*6)
    """
    assert len(trans_and_poses.shape) == 3
    valid_trans = trans_and_poses[:, :, :3]
    valid_poses = rectify_rot6d(trans_and_poses[:, :, 3:])
    return torch.cat([valid_trans, valid_poses], dim=2)


def geodesic_loss(input, target, reduction='none', eps=1e-7):
    # input.shape == [B, 3, 3]
    # target.shape == [B, 3, 3]
    r_diffs = input @ target.permute(0, 2, 1)
    traces = r_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))
    if reduction == 'sum':
        return dists.sum()
    elif reduction == 'mean':
        return dists.mean()
    assert reduction == 'none'
    return dists  # [B]


@dataclass
class MotionPretrainCriterionConfig(FairseqDataclass):
    motion_gt_keep_prob: float = field(
        default=1.0, metadata={"help": "the probability to keep using ground-truth tokens as input when training"}
    )


@register_criterion(
    "motion_pretrain_criterion", dataclass=MotionPretrainCriterionConfig
)
class MotionPretrainCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            motion_gt_keep_prob,
    ):
        super().__init__(task)
        self.motion_gt_keep_prob = motion_gt_keep_prob

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert not isinstance(sample, list)

        net_input = sample["net_input"]
        loss, logging_output, output_pose = self.compute_loss(model, sample, net_input, tag='v1_')

        sample_size = 1
        logging_output["ntokens"] = sample["ntokens"]
        logging_output["nsentences"] = sample["nsentences"]
        logging_output["sample_size"] = sample_size

        if self.motion_gt_keep_prob < 1.:
            gt_pose = net_input["prev_output_pose"]
            bsz, slen, edim = gt_pose.shape
            corrupt_pose = torch.where(
                torch.lt(torch.rand(bsz, slen - 1, 1, dtype=gt_pose.dtype, device=gt_pose.device).repeat(1, 1, edim),
                         self.motion_gt_keep_prob),
                gt_pose[:, 1:],
                rectify_poses(output_pose.detach()[:, :-1]),
            )
            corrupt_pose = torch.cat([gt_pose[:, :1], corrupt_pose], dim=1)
            assert corrupt_pose.shape == (bsz, slen, edim)

            corrupt_input = {}
            for k, v in net_input.items():
                corrupt_input[k] = v
            corrupt_input["prev_output_pose"] = corrupt_pose

            loss_v2, logging_output_v2, _ = self.compute_loss(model, sample, corrupt_input, tag='v2_')
            loss = (loss + loss_v2) * 0.5
            logging_output.update(logging_output_v2)

        logging_output["loss"] = loss.data
        return loss, sample_size, logging_output

    def compute_loss(self, model, sample, net_input, tag=''):

        target_mask = sample["target_mask"]
        target_pose = sample["target_pose"]
        sample_size = sample["ntokens"]

        bsz, slen, pose_param_dim = target_pose.shape
        num_joints = (pose_param_dim - 3) // 6

        output_pose = model(**net_input)[0]

        loc_loss = torch.sum(
            target_mask * F.l1_loss(input=output_pose[:, :, :3],
                                    target=target_pose[:, :, :3],
                                    reduction='none').mean(dim=-1)
        ) / sample_size * 50.

        output_rot = rot6d_to_rotmat(output_pose[:, :, 3:]).view(bsz * slen * num_joints, 3, 3)
        target_rot = rot6d_to_rotmat(target_pose[:, :, 3:]).view(bsz * slen * num_joints, 3, 3)

        rot_mse_loss = torch.sum(
            target_mask * F.mse_loss(input=output_rot,
                                     target=target_rot,
                                     reduction='none').view(bsz, slen, num_joints * 9).mean(dim=-1)
        ) / sample_size * 10.

        rot_geo_loss = torch.sum(
            target_mask * geodesic_loss(input=output_rot,
                                        target=target_rot,
                                        reduction='none').view(bsz, slen, num_joints).mean(dim=-1)
        ) / sample_size

        loss = loc_loss + rot_mse_loss + rot_geo_loss
        logging_output = {
            tag + "loc_loss": loc_loss.data,
            tag + "rot_mse_loss": rot_mse_loss.data,
            tag + "rot_geo_loss": rot_geo_loss.data,
        }
        return loss, logging_output, output_pose

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )

        for k in logging_outputs[0].keys():
            if k.endswith('_loss'):
                metrics.log_scalar(
                    k, sum(log.get(k, 0) for log in logging_outputs) / sample_size, sample_size, round=3
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
