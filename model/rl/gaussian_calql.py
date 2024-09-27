"""
Calibrated Conservative Q-Learning (CalQL) for Gaussian policy.

"""

import torch
import torch.nn as nn
import logging
from copy import deepcopy
import numpy as np

from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)


class CalQL_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        cql_clip_diff_min=-np.inf,
        cql_clip_diff_max=np.inf,
        cql_min_q_weight=5.0,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_min_q_weight = cql_min_q_weight

        # initialize critic networks
        self.critic = critic.to(self.device)
        self.target_critic = deepcopy(critic).to(self.device)

        # initialize target networks
        self.actor = actor.to(self.device)

    def loss_critic(
        self, obs, next_obs, actions, random_actions, rewards, returns, dones, gamma
    ):
        # for some reason, alpha is not used in the critic loss

        # Get initial TD loss
        q_data1, q_data2 = self.critic(obs, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.forward(
                next_obs, deterministic=False, get_logprob=True
            )
            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            target_q1 = rewards + gamma * (1 - dones) * next_q1
            target_q2 = rewards + gamma * (1 - dones) * next_q2
        td_loss_1 = nn.functional.mse_loss(q_data1, target_q1)
        td_loss_2 = nn.functional.mse_loss(q_data2, target_q2)

        # Get actions and logprobs
        log_rand_pi = 0.5 ** random_actions.shape[-1]
        pi_actions, log_pi = self.forward(
            obs, deterministic=False, reparameterize=False, get_logprob=True
        )

        # Random action Q values. Since the number of samples is small, we loop over the samples
        # to avoid complicated dictionary reshaping
        q_rand_1_list = []
        q_rand_2_list = []
        for a in range(random_actions.shape[0]):
            q_rand_1, q_rand_2 = self.critic(obs, random_actions[a])
            q_rand_1 = q_rand_1 - log_rand_pi
            q_rand_2 = q_rand_2 - log_rand_pi
            q_rand_1_list.append(q_rand_1)
            q_rand_2_list.append(q_rand_2)
        q_rand_1 = torch.stack(q_rand_1_list, dim=0)  # (num_samples, batch_size)
        q_rand_2 = torch.stack(q_rand_2_list, dim=0)  # (num_samples, batch_size)

        # Policy action Q values
        q_pi_1, q_pi_2 = self.critic(obs, pi_actions)
        q_pi_1 = q_pi_1 - log_pi
        q_pi_2 = q_pi_2 - log_pi

        # Ensure calibration w.r.t. value function estimate
        q_pi_1 = torch.max(q_pi_1, returns)[None]  # (1, batch_size)
        q_pi_2 = torch.max(q_pi_2, returns)[None]  # (1, batch_size)
        cat_q_1 = torch.cat([q_rand_1, q_pi_1], dim=0)  # (num_samples+1, batch_size)
        cql_qf1_ood = torch.logsumexp(cat_q_1, dim=0)  # sum over num_samples
        cat_q_2 = torch.cat([q_rand_2, q_pi_2], dim=0)  # (num_samples+1, batch_size)
        cql_qf2_ood = torch.logsumexp(cat_q_2, dim=0)  # sum over num_samples

        # Subtract the log likelihood of the data
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q_data1,
            min=self.cql_clip_diff_min,
            max=self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q_data2,
            min=self.cql_clip_diff_min,
            max=self.cql_clip_diff_max,
        ).mean()

        cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
        cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight

        # Sum the two losses
        critic_loss = td_loss_1 + td_loss_2 + cql_min_qf1_loss + cql_min_qf2_loss

        return critic_loss

    def loss_actor(self, obs, alpha):
        new_actions, log_probs = self.forward(
            obs, deterministic=False, get_logprob=True
        )
        q1, q2 = self.critic(obs, new_actions)
        q = torch.min(q1, q2)
        actor_loss = -torch.mean(q - alpha * log_probs)
        return actor_loss

    def loss_temperature(self, obs, alpha, target_entropy):
        _, logprob = self.forward(
            obs,
            deterministic=False,
            reparameterize=True,
            get_logprob=True,
        )
        loss_alpha = -torch.mean(alpha * (logprob.detach() + target_entropy))
        return loss_alpha

    def update_target_critic(self, tau):
        # copy all params from critic to target_critic with tau learning rate
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # ---------- Sampling ----------#

    def forward(
        self,
        cond,
        deterministic=False,
        reparameterize=False,
        get_logprob=False,
    ):
        return super().forward(
            cond=cond,
            deterministic=deterministic,
            reparameterize=reparameterize,
            get_logprob=get_logprob,
        )
