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

@dataclass
class TargetRegressionCriterionConfig(FairseqDataclass):
    motion_gt_keep_prob: float = field(
        default=1.0, metadata={"help": "the probability to keep using ground-truth tokens as input when training"}
    )


@register_criterion(
    "target_regression_criterion", dataclass=TargetRegressionCriterionConfig
)
class TargetRegressionCriterion(FairseqCriterion):
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

        net_output = model(**sample["net_input"])
        #loss, nll_loss, ntokens = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)
        loss, nll_loss, ntokens = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)
        #sample_size = (sample["target"].size(0) if self.sentence_avg else ntokens)
        sample_size = ntokens

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, net_input, tag=''):

        target_pred = net_output[0]
        target_mask = sample["target_mask"]
        target_length = sample["target_length"]
        target = sample["target"]
        ntokens = sample["ntokens"]

        bsz, tgt_length, dim_a = target.shape

        sample_size = max(ntokens, 1.0)

        mse_loss = torch.sum(target_mask * F.mse_loss(input=target_pred, target=target, reduction='none').mean(dim=-1)) / sample_size   # .view(bsz, tgt_length, dim_a)
        loss = mse_loss

        nll_loss = torch.tensor(0.0)

        return loss, nll_loss, ntokens

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
