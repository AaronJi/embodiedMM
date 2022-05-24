import torch

class ActionPredictionCriterion(object):
    def __init__(self):

        return

    def cal_loss(self, model, sample):

        states, actions, rewards, dones, rtg, timesteps, attention_mask = sample
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = model.forward(states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,)

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2)
        loss = loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        sample_size = None
        logging_output = None
        return loss, sample_size, logging_output
