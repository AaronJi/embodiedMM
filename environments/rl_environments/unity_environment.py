import gym
import numpy as np
import torch
import argparse


### definitions
bodyparts = [
    'hips',
    'chest',
    'spine',
    'head',
    'thighL',
    'shinL',
    'footL',
    'thighR',
    'shinR',
    'footR',
    'upperarmL',
    'forarmL',
    'handL',
    'upperarmR',
    'forarmR',
    'handR'
]
n_bodyparts = len(bodyparts)
not_active_bodyparts = ['hips', 'handL', 'handR'] # bodyparts without proprioceptive and actions
action_dim2_bodyparts = ['head', 'thighL', 'thighR', 'upperarmL', 'upperarmR']  # bodyparts with only 2 dim actions
action_dim1_bodyparts = ['shinL', 'shinR', 'forarmL', 'forarmR']  # bodyparts with only 1 dim actions

dim_obs_orginal = 243
dim_act_orginal = 39
dim_exteroceptive = 1 + 3 + 3 + 4 + 4 + 3  # 18
dim_bodypart_proprioceptive = 1 + 3 + 3 + 3 + 4 + 1  # 15
dim_bodypart_proprioceptive_not_for_leaf = 4 + 1  # 5
dim_bodypart_act = 3 + 1  # 4

### conversion for obs
assert dim_obs_orginal == dim_exteroceptive + n_bodyparts*dim_bodypart_proprioceptive - len(not_active_bodyparts)*dim_bodypart_proprioceptive_not_for_leaf
dim_proprioceptive = n_bodyparts*dim_bodypart_proprioceptive # 240

## permute and mask
permute_proprioceptive = torch.zeros(dim_obs_orginal, dim_proprioceptive)
permute_proprioceptive_back = torch.zeros(dim_proprioceptive, dim_obs_orginal)
obs_mask = torch.ones(1, dim_proprioceptive)

i = 0
j = 0
for bodypart in bodyparts:
    if bodypart not in not_active_bodyparts:
        for k in range(dim_bodypart_proprioceptive):
            permute_proprioceptive[dim_exteroceptive+i+k, j+k] = 1
            permute_proprioceptive_back[j+k, i+k] = 1
            obs_mask[:, j+k] = 0
        # print('%s NOT LEAF, set permute matrix with i from %i to %i, j from %i to %i to 1.' % (bodypart, i, i + dim_bodypart_proprioceptive, j, j + dim_bodypart_proprioceptive))
        i += dim_bodypart_proprioceptive
    else:
        for k in range(dim_bodypart_proprioceptive-dim_bodypart_proprioceptive_not_for_leaf):
            permute_proprioceptive[dim_exteroceptive+i+k, j+k] = 1
            permute_proprioceptive_back[j + k, i + k] = 1
            obs_mask[:, j+k] = 0
        # print('%s IS LEAF, set permute matrix with i from %i to %i, j from %i to %i to 1, keep j with 0 until %i' % (bodypart, i, i + dim_bodypart_proprioceptive-dim_bodypart_proprioceptive_not_for_leaf, j, j + dim_bodypart_proprioceptive-dim_bodypart_proprioceptive_not_for_leaf, j + dim_bodypart_proprioceptive))
        i += dim_bodypart_proprioceptive-dim_bodypart_proprioceptive_not_for_leaf
    j += dim_bodypart_proprioceptive

### conversion for action
assert dim_act_orginal == (n_bodyparts-len(not_active_bodyparts))*dim_bodypart_act - len(action_dim2_bodyparts)*1 - len(action_dim1_bodyparts)*2
dim_act = n_bodyparts*dim_bodypart_act

## permute and mask
permute_act = torch.zeros(dim_act_orginal, dim_act)
permute_act_back = torch.zeros(dim_act, dim_act_orginal)
act_mask = torch.ones(1, dim_act)

# chest
permute_act[0, 4] = 1
permute_act[1, 5] = 1
permute_act[2, 6] = 1
permute_act[26, 7] = 1

permute_act_back[4, 0] = 1
permute_act_back[5, 1] = 1
permute_act_back[6, 2] = 1
permute_act_back[7, 26] = 1

# spine
permute_act[3, 8] = 1
permute_act[4, 9] = 1
permute_act[5, 10] = 1
permute_act[27, 11] = 1

permute_act_back[8, 3] = 1
permute_act_back[9, 4] = 1
permute_act_back[10, 5] = 1
permute_act_back[11, 27] = 1

# head
permute_act[24, 12] = 1
permute_act[25, 13] = 1

permute_act[28, 15] = 1

permute_act_back[12, 24] = 1
permute_act_back[13, 25] = 1

permute_act_back[15, 28] = 1

# thighL
permute_act[6, 16] = 1
permute_act[7, 17] = 1

permute_act[29, 19] = 1

permute_act_back[16, 6] = 1
permute_act_back[17, 7] = 1

permute_act_back[19, 29] = 1

# thinL
permute_act[10, 20] = 1


permute_act[30, 23] = 1

permute_act_back[20, 10] = 1


permute_act_back[23, 30] = 1

# footL
permute_act[15, 24] = 1
permute_act[16, 25] = 1
permute_act[17, 26] = 1
permute_act[31, 27] = 1

permute_act_back[24, 15] = 1
permute_act_back[25, 16] = 1
permute_act_back[26, 17] = 1
permute_act_back[27, 31] = 1

# thighR
permute_act[8, 28] = 1
permute_act[9, 29] = 1

permute_act[32, 31] = 1

permute_act_back[28, 8] = 1
permute_act_back[29, 9] = 1

permute_act_back[31, 32] = 1

# thinR
permute_act[11, 32] = 1


permute_act[33, 35] = 1

permute_act_back[32, 11] = 1


permute_act_back[35, 33] = 1

# footR
permute_act[12, 36] = 1
permute_act[13, 37] = 1
permute_act[14, 38] = 1
permute_act[34, 39] = 1

permute_act_back[36, 12] = 1
permute_act_back[37, 13] = 1
permute_act_back[38, 14] = 1
permute_act_back[39, 34] = 1

# upperarmL
permute_act[18, 40] = 1
permute_act[19, 41] = 1

permute_act[35, 43] = 1

permute_act_back[40, 18] = 1
permute_act_back[41, 19] = 1

permute_act_back[43, 35] = 1

# forarmL
permute_act[22, 44] = 1


permute_act[36, 47] = 1

permute_act_back[44, 22] = 1


permute_act_back[47, 36] = 1

# upperarmR
permute_act[20, 52] = 1
permute_act[21, 53] = 1

permute_act[37, 55] = 1

permute_act_back[52, 20] = 1
permute_act_back[53, 21] = 1

permute_act_back[55, 37] = 1

# forarmR
permute_act[23, 56] = 1


permute_act[38, 59] = 1

permute_act_back[56, 23] = 1


permute_act_back[59, 38] = 1


# act mask
j = 0
for bodypart in bodyparts:
    if bodypart not in not_active_bodyparts:
        if bodypart in action_dim2_bodyparts:
            act_mask[:, j] = 0
            act_mask[:, j + 1] = 0
            act_mask[:, j + 2] = 1
            act_mask[:, j + 3] = 0
        elif bodypart in action_dim1_bodyparts:
            act_mask[:, j] = 0
            act_mask[:, j + 1] = 1
            act_mask[:, j + 2] = 1
            act_mask[:, j + 3] = 0
        else:
            act_mask[:, j] = 0
            act_mask[:, j + 1] = 0
            act_mask[:, j + 2] = 0
            act_mask[:, j + 3] = 0
    j += dim_bodypart_act


def convert_observation(obs_original: torch.Tensor) -> dict:
    assert dim_obs_orginal == obs_original.shape[1]  # 243
    exteroceptive = obs_original[:, :dim_exteroceptive]
    proprioceptive = torch.mm(obs_original, permute_proprioceptive)  # shape = [obs.shape[0], dim_bodypart_proprioceptive]

    obs_mask_batch = obs_mask.repeat(obs_original.shape[0], 1)
    obs = {'proprioceptive': proprioceptive, 'exteroceptive': exteroceptive, 'obs_mask': obs_mask_batch}
    return obs


def convert_observation_back(obs: dict) -> torch.Tensor:
    proprioceptive = obs['proprioceptive']
    exteroceptive = obs['exteroceptive']
    obs_original = torch.cat([exteroceptive, torch.mm(proprioceptive, permute_proprioceptive_back)], dim=1)
    return obs_original


def convert_action(action_original: torch.Tensor) -> torch.Tensor:
    assert dim_act_orginal == action_original.shape[1]  # 39
    action = torch.mm(action_original, permute_act)
    return action


def convert_action_back(action: torch.Tensor) -> torch.Tensor:
    action_original = torch.mm(action, permute_act_back)
    return action_original



'''
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
def make_unity_env():
    walker_env_path = '/Users/jiluo-wendu/Projects/unity/my_single_walker_env.app'
    unity_env = UnityEnvironment(file_name=walker_env_path, seed=1, side_channels=[])
    env = UnityToGymWrapper(unity_env, uint8_visual=False, flatten_branched=False, allow_multiple_obs=False, action_space_seed=0)
    return env
'''



class UnityMLEnvironment(object):
    def __init__(self, cfg: argparse.Namespace, env_name):
        self.cfg = cfg
        self.name = env_name

        if env_name == 'Walker':
            self.state_dim = 243
            self.action_dim = 39

            #self.state_bounds = [env.observation_space.low, env.observation_space.high]
            #self.action_bounds = [env.action_space.low, env.action_space.high]
            #self.reward_bounds = [env.reward_range[0], env.reward_range[1]]

            print('initialize environment %s with state dim = %i and action dim = %i' % (self.name, self.state_dim, self.action_dim))
        else:
            raise NotImplementedError
        return

    def reset(self):
        pass

    def step(self, action):
        pass
