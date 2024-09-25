"""
Imitation Bootstrapped Reinforcement Learning (IBRL) for Gaussian policy.

"""

import torch
import torch.nn as nn
import logging
from copy import deepcopy

from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)


class IBRL_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        n_critics,
        actor_explore_std,
        critic_clip_c,
        soft_action_sample=False,
        soft_action_sample_beta=0.1,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)

        self.actor_explore_std = actor_explore_std
        self.critic_clip_c = critic_clip_c
        self.soft_action_sample = soft_action_sample
        self.soft_action_sample_beta = soft_action_sample_beta

        self.target_actor = deepcopy(actor)

        self.imitation_policy = deepcopy(actor)
        for param in self.imitation_policy.parameters():
            param.requires_grad = False

        # initialize critic networks
        self.critic_networks = [
            deepcopy(critic).to(self.device) for _ in range(n_critics)
        ]
        self.critic_networks = nn.ModuleList(self.critic_networks)

        # initialize target networks
        self.target_networks = [
            deepcopy(critic).to(self.device) for _ in range(n_critics)
        ]
        self.target_networks = nn.ModuleList(self.target_networks)

    def get_random_indices(self, sz=None, num_ind=2):
        # get num_ind random indices from a set of size sz (used for getting critic targets)
        if sz is None:
            sz = len(self.critic_networks)
        perm = torch.randperm(sz)
        ind = perm[:num_ind].to(self.device)
        return ind

    def loss_critic(self, obs, next_obs, actions, rewards, dones, gamma):
        # get random critic index
        critic_ind = self.get_random_indices()
        q1_ind = critic_ind[0]
        q2_ind = critic_ind[1]

        with torch.no_grad():
            # get the next IL action
            next_actions_il = super().forward(
                cond=next_obs,
                deterministic=False,
                network_override=self.imitation_policy,
            )

            # get the next RL action
            next_actions_rl = self.forward(
                cond=next_obs,
                deterministic=True,  # note determinstic=True here so we can clip the noise
            )
            exploration_noise = (
                torch.randn_like(next_actions_rl) * self.actor_explore_std
            )
            exploration_noise = torch.clamp(
                exploration_noise, -self.critic_clip_c, self.critic_clip_c
            )

            next_actions_rl = next_actions_rl + exploration_noise

            # get the IL Q value
            next_q1_il = self.target_networks[q1_ind](next_obs, next_actions_il)[0]
            next_q2_il = self.target_networks[q2_ind](next_obs, next_actions_il)[0]
            next_q_il = torch.min(next_q1_il, next_q2_il)

            # get the RL Q value
            next_q1_rl = self.target_networks[q1_ind](next_obs, next_actions_rl)[0]
            next_q2_rl = self.target_networks[q2_ind](next_obs, next_actions_rl)[0]
            next_q_rl = torch.min(next_q1_rl, next_q2_rl)

            next_q = torch.where(next_q_il > next_q_rl, next_q_il, next_q_rl)

            # terminal state mask
            mask = 1 - dones

            # flatten
            rewards = rewards.view(-1)
            next_q = next_q.view(-1)
            mask = mask.view(-1)

            # target value
            target_q = rewards + gamma * next_q * mask  # (B,)

        # loop over all critic networks and compute value estimate
        current_q = [critic(obs, actions)[0] for critic in self.critic_networks]
        current_q = torch.stack(current_q, dim=-1)  # (B, n_critics)
        loss_critic = torch.mean((current_q - target_q.unsqueeze(-1)) ** 2)
        return loss_critic

    def loss_actor(self, obs):
        # compute current action and entropy
        action = self.forward(obs, deterministic=False, reparameterize=True)

        # loop over all critic networks and compute value estimate
        current_q = [critic(obs, action)[0] for critic in self.critic_networks]
        current_q = torch.stack(current_q, dim=-1)  # (B, n_critics)

        loss_actor = -torch.min(current_q).mean()  # min over critics, mean over batch
        return loss_actor

    def update_target_critic(self, tau):
        for target_param, source_param in zip(
            self.target_networks.parameters(), self.critic_networks.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    def update_target_actor(self, tau):
        for target_param, source_param in zip(
            self.target_actor.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    # ---------- Sampling ----------#

    def forward(
        self,
        cond,
        deterministic=False,
        reparameterize=False,  # allow gradient
    ):
        critic_ind = self.get_random_indices()
        q1_ind = critic_ind[0]
        q2_ind = critic_ind[1]

        # sample an action from the imitation policy
        imitation_action = super().forward(
            cond=cond,
            deterministic=deterministic,
            network_override=self.imitation_policy,
        )

        # sample an action from the RL policy
        rl_action = super().forward(
            cond=cond,
            deterministic=deterministic,
            reparameterize=reparameterize,
        )

        # compute Q value of imitation policy
        q_imitation_1 = self.critic_networks[q1_ind](cond, imitation_action)[0]
        q_imitation_2 = self.critic_networks[q2_ind](cond, imitation_action)[0]
        q_imitation = torch.min(q_imitation_1, q_imitation_2)

        # compute Q value of RL policy
        q_rl_1 = self.critic_networks[q1_ind](cond, rl_action)[0]
        q_rl_2 = self.critic_networks[q2_ind](cond, rl_action)[0]
        q_rl = torch.min(q_rl_1, q_rl_2)

        if self.soft_action_sample:
            # compute the Q weights with probability proportional to exp(\beta * Q(a))
            qw_il = torch.exp(q_imitation * self.soft_action_sample_beta)
            qw_rl = torch.exp(q_rl * self.soft_action_sample_beta)
            q_weights = torch.softmax(torch.stack([qw_il, qw_rl], dim=-1), dim=-1)

            # sample indices according to the weights
            q_indices = torch.multinomial(q_weights, 1)

            # repeat the q_indices to match the action shape
            q_indices = q_indices[:, None, None].repeat(
                1, imitation_action.shape[-2], imitation_action.shape[-1]
            )

            # select the action with higher Q value
            action = torch.where(q_indices == 0, imitation_action, rl_action)
        else:
            # repeat the q values to match the action shape
            q_imitation = q_imitation[:, None, None].repeat(
                1, imitation_action.shape[-2], imitation_action.shape[-1]
            )
            q_rl = q_rl[:, None, None].repeat(
                1, rl_action.shape[-2], rl_action.shape[-1]
            )

            # select the action with higher Q value
            action = torch.where(q_imitation > q_rl, imitation_action, rl_action)

        return action
