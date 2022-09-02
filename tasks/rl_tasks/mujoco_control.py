import argparse
import os
import logging
import random
import numpy as np
import torch

from fairseq.data import Dictionary
from tasks.ofa_task import OFATask, OFAConfig
from environments.rl_environments.gym_environment import GymEnvironment
from environments.rl_environments.unity_environment import UnityMLEnvironment
from data.rl_data.gym_original_dataset import GymDataset
from data.rl_data.unity_dataset import UnityDataset
from models.trajectory.decision_transformer import DecisionTransformer
from models.trajectory.mlp_bc import MLPBCModel
from models.trajectory.trajectory_OFA import TrajectoryOFAModel
from criterions.action_pred_error import ActionPredictionCriterion
from criterions.label_smoothed_cross_entropy import AdjustLabelSmoothedCrossEntropyCriterion
from utils.rl.rl_eval_utils import evaluate_episode, evaluate_episode_rtg

logger = logging.getLogger(__name__)

class MujocoControlTask(OFATask):
    def __init__(self, cfg: argparse.Namespace):
        self.name = 'mujoco_control'

        """Setup the task."""
        # load dictionaries
        #src_dict = super().load_dictionary(os.path.join(cfg.bpe_dir, "dict.txt"))
        #tgt_dict = super().load_dictionary(os.path.join(cfg.bpe_dir, "dict.txt"))

        src_dict = Dictionary()
        tgt_dict = Dictionary()

        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        #for i in range(OFAConfig.code_dict_size):
        #    src_dict.add_symbol("<code_{}>".format(i))
        #    tgt_dict.add_symbol("<code_{}>".format(i))

        # quantization
        for i in range(OFAConfig.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))

        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))

        super().__init__(OFAConfig, src_dict, tgt_dict)
        self.cfg = cfg

        self.model = None
        self.criterion = None

        self.device = self.cfg.device
        self.K = self.cfg.K

        return

    def build(self):
        if self.cfg.env == 'hopper':
            self.env_name = 'Hopper-v3'
            self.max_ep_len = 1000
            self.env_targets = [3600, 1800]  # evaluation conditioning targets
            self.scale = 1000.  # normalization for rewards/returns
            self.env = GymEnvironment(self.cfg, self.env_name)
            self.dataset = GymDataset(self.cfg, self.env, self.max_ep_len, self.scale)
            self.dataset_dir = 'dataset/gym_data/'
            self.dataset_name = f'{self.cfg.env}-{self.cfg.dataset}-v2.pkl'
            self.exp_prefix = 'gym-experiment'
        elif self.cfg.env == 'halfcheetah':
            self.env_name = 'HalfCheetah-v3'
            self.max_ep_len = 1000
            self.env_targets = [12000, 6000]
            self.scale = 1000.
            self.env = GymEnvironment(self.cfg, self.env_name)
            self.dataset = GymDataset(self.cfg, self.env, self.max_ep_len, self.scale)
            self.dataset_dir = 'dataset/gym_data/'
            self.dataset_name = f'{self.cfg.env}-{self.cfg.dataset}-v2.pkl'
            self.exp_prefix = 'gym-experiment'
        elif self.cfg.env == 'walker2d':
            self.env_name = 'Walker2d-v3'
            self.max_ep_len = 1000
            self.env_targets = [5000, 2500]
            self.scale = 1000.
            self.env = GymEnvironment(self.cfg, self.env_name)
            self.dataset = GymDataset(self.cfg, self.env, self.max_ep_len, self.scale)
            self.dataset_dir = 'dataset/gym_data/'
            self.dataset_name = f'{self.cfg.env}-{self.cfg.dataset}-v2.pkl'
            self.exp_prefix = 'gym-experiment'
        elif self.cfg.env == 'ant':
            self.env_name = 'Ant-v3'
            self.max_ep_len = 1000
            self.env_targets = [12000, 6000]  # TODO
            self.scale = 1000.
            self.env = GymEnvironment(self.cfg, self.env_name)
            self.dataset = GymDataset(self.cfg, self.env, self.max_ep_len, self.scale)
            self.dataset_dir = 'dataset/gym_data/'
            self.dataset_name = f'{self.cfg.env}-{self.cfg.dataset}-v2.pkl'
            self.exp_prefix = 'gym-experiment'
        elif self.cfg.env == 'reacher2d':
            self.env_name = 'Reacher2d'
            self.max_ep_len = 100
            self.env_targets = [76, 40]
            self.scale = 10.
            self.env = GymEnvironment(self.cfg, self.env_name)
            self.dataset = GymDataset(self.cfg, self.env, self.max_ep_len, self.scale)
            self.dataset_dir = 'dataset/gym_data/'
            self.dataset_name = f'{self.cfg.env}-{self.cfg.dataset}-v2.pkl'
            self.exp_prefix = 'gym-experiment'
        elif self.cfg.env == 'walker':
            self.env_name = 'Walker'
            self.max_ep_len = 1000
            self.env_targets = [1000, 500]
            self.scale = 1000.
            self.env = UnityMLEnvironment(self.cfg, self.env_name)
            self.dataset = UnityDataset(self.cfg, self.env, self.max_ep_len, self.scale)
            self.dataset_dir = 'dataset/unity_data/'
            self.dataset_name = f'{self.cfg.env}.pkl'
            self.exp_prefix = 'unity-experiment'
        else:
            raise NotImplementedError

        self.dataset_path = self.dataset_dir + self.dataset_name

        if self.cfg.model == 'bc':
            self.env_targets = self.env_targets[:1]  # since BC ignores target, no need for different evaluations

        return

    def load_dataset(self):

        dataset = self.cfg.dataset

        group_name = f'{self.exp_prefix}-{self.env_name}-{dataset}'
        self.exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

        # load dataset

        self.dataset.load(self.dataset_path)

        print('=' * 50)
        print(f'Starting new experiment: {self.env.name} {dataset}')
        print(f'{len(self.dataset.traj_lens)} trajectories, {self.dataset.num_timesteps} timesteps found')
        print(f'Average return: {np.mean(self.dataset.returns):.2f}, std: {np.std(self.dataset.returns):.2f}')
        print(f'Max return: {np.max(self.dataset.returns):.2f}, min: {np.min(self.dataset.returns):.2f}')
        print('=' * 50)

        return

    def build_model(self, model_type):
        self.model_type = model_type
        if model_type == 'dt':
            model = DecisionTransformer(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                max_length=self.K,
                max_ep_len=self.max_ep_len,
                hidden_size=self.cfg.embed_dim,
                n_layer=self.cfg.n_layer,
                n_head=self.cfg.n_head,
                n_inner=4 * self.cfg.embed_dim,
                activation_function=self.cfg.activation_function,
                #n_positions=1024,
                n_ctx=1024,
                resid_pdrop=self.cfg.dropout,
                attn_pdrop=self.cfg.dropout,
            )
        elif model_type == 'bc':
            model = MLPBCModel(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                max_length=self.K,
                hidden_size=self.cfg.embed_dim,
                n_layer=self.cfg.n_layer,
            )
        elif model_type == 'ofa':
            model = TrajectoryOFAModel(self.cfg, self)
        else:
            raise NotImplementedError

        model = model.to(device=self.device)
        self.model = model
        return self.model

    def build_criterion(self, criterion_type):
        self.criterion_type = criterion_type
        if criterion_type == 'action_pred_error':
            self.criterion = ActionPredictionCriterion()
        elif criterion_type == 'label_smoothed_cross_entropy':
            from criterions.label_smoothed_cross_entropy import AdjustLabelSmoothedCrossEntropyCriterionConfig
            self.criterion = AdjustLabelSmoothedCrossEntropyCriterion(self, AdjustLabelSmoothedCrossEntropyCriterionConfig.sentence_avg, AdjustLabelSmoothedCrossEntropyCriterionConfig.label_smoothing)
        else:
            raise NotImplementedError
        return self.criterion

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()

        if self.criterion_type == 'action_pred_error':
            loss, sample_size, logging_output = criterion.cal_loss(model, sample)
        else:
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0

        #print(loss)
        #exit(4)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
        optimizer.step()

        return loss.detach().cpu().item()

    def valid_step(self):
        num_eval_episodes = self.cfg.num_eval_episodes

        def eval_episodes(target_rew):

            def fn(model):
                returns, lengths = [], []
                for _ in range(num_eval_episodes):
                    with torch.no_grad():
                        if self.model_type == 'dt':
                            ret, length = evaluate_episode_rtg(
                                self.env,
                                model,
                                max_ep_len=self.max_ep_len,
                                scale=self.scale,
                                target_return=target_rew / self.scale,
                                mode=self.cfg.mode,
                                state_mean=self.dataset.state_mean,
                                state_std=self.dataset.state_std,
                                device=self.device,
                            )
                        else:
                            ret, length = evaluate_episode(
                                self.env,
                                model,
                                max_ep_len=self.max_ep_len,
                                target_return=target_rew / self.scale,
                                mode=self.cfg.mode,
                                state_mean=self.dataset.state_mean,
                                state_std=self.dataset.state_std,
                                device=self.device,
                            )
                    returns.append(ret)
                    lengths.append(length)
                return {
                    f'target_{target_rew}_return_mean': np.mean(returns),
                    f'target_{target_rew}_return_std': np.std(returns),
                    f'target_{target_rew}_length_mean': np.mean(lengths),
                    f'target_{target_rew}_length_std': np.std(lengths),
                }

            return fn

        eval_fns = [eval_episodes(tar) for tar in self.env_targets]
        outputs_list = []
        for eval_fn in eval_fns:
            outputs = eval_fn(self.model)
            outputs_list.append(outputs)
        return outputs_list
