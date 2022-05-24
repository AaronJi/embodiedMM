
import argparse
import gym

class GymEnvironment(object):
    def __init__(self, cfg: argparse.Namespace, env_name):
        self.cfg = cfg
        self.name = env_name

        if env_name == 'Reacher2d':
            from environments.rl_environments.reacher_2d import Reacher2dEnv
            env = Reacher2dEnv()
        else:
            env = gym.make(self.name)

        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.env = env
        print('initialize environment %s with state dim = %i and action dim = %i' % (self.name, self.state_dim, self.act_dim))
        return

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
