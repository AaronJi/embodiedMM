import torch

class ActionPredictionCriterion(object):
    def __init__(self):

        return

    def cal_loss(self, model, sample):

        states, actions, rewards, dones, rtg, timesteps, attention_mask = sample
        actions_target = torch.clone(actions)

        states_pred, actions_pred, returns_pred = model.forward(states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,)

        action_dim = actions_pred.shape[2]
        action_pred = actions_pred.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        action_target = actions_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]

        loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2)
        loss = loss_fn(
            None, action_pred, None,
            None, action_target, None,
        )

        sample_size = None
        logging_output = None

        return loss, sample_size, logging_output
