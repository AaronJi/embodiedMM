import argparse
import logging
import torch
import time
import numpy as np

logger = logging.getLogger(__name__)

class TrainerLight(object):
    def __init__(self, cfg: argparse.Namespace, task, model, criterion, quantizer):

        self.cfg = cfg
        self.task = task
        self.model = model
        self.criterion = criterion
        self.quantizer = quantizer

        self.batch_size = cfg.batch_size

        self.diagnostics = dict()
        self.start_time = time.time()
        return

    def _build_optimizer(self):

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        warmup_steps = self.cfg.warmup_steps
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps + 1) / warmup_steps, 1)
        )
        return

    def zero_grad(self):
        self.optimizer.zero_grad()

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        logger.info("begin training epoch {}".format(epoch))

        self.lr_step_begin_epoch(epoch)

        if self.quantizer is not None:
            self.quantizer.begin_epoch(epoch)

        # task specific setup per epoch
        self.task.begin_epoch(epoch, self.model)

    def _prepare_sample(self, sample):
        return sample, False

    def train_step(self, samples, raise_oom=False):
        self.model.train()
        #self.criterion.train()
        self.zero_grad()

        scale_way = 'normalize'
        if self.cfg.model == 'ofa':
            scale_way = 'standardize'
        sample = self.task.dataset.get_batch(self.batch_size, scale_way=scale_way)
        sample, is_dummy_batch = self._prepare_sample(sample)

        loss = self.task.train_step(sample, self.model, self.criterion, self.optimizer)
        return loss

    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""

        self.model.eval()
        outputs = self.task.valid_step()

        return outputs

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step(None, False)
            train_losses.append(train_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        #with torch.no_grad():
        #    self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        logs['time/training'] = time.time() - train_start

        #print(train_losses)
        #print(exit(3))
        eval_start = time.time()

        self.model.eval()
        outputs_list = self.valid_step(None, raise_oom=False)
        for outputs in outputs_list:
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def eval(self, print_logs=False):
        self.model.eval()

        logs = dict()

        outputs_list = self.valid_step(None, raise_oom=False)
        for outputs in outputs_list:
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            for k, v in logs.items():
                print(f'{k}: {v}')

        return outputs_list

    def lr_step_begin_epoch(self, epoch):
        """Adjust the learning rate at the beginning of the epoch."""
        self.lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()


