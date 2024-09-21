"""
Imitation bootstrapped reinforcement learning (IBRL) for Gaussian policy.

"""

import torch
import torch.nn as nn
import logging
from model.common.gaussian import GaussianModel
from copy import deepcopy

import torch.distributions as D
from util.network import soft_update


log = logging.getLogger(__name__)


class IBRL_Gaussian(GaussianModel):

    def __init__(
        self,
        actor,
        critic,
        n_critics,
        exploration_noise=0.1,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)

        # initialize critic networks
        self.critic_networks = []
        for i in range(n_critics):
            critic_copy = deepcopy(critic)
            critic_copy = critic_copy.to(self.device)
            self.critic_networks.append(critic_copy)
        self.critic_networks = nn.ModuleList(self.critic_networks)

        # initialize target networks
        self.target_networks = []
        for i in range(n_critics):
            critic_copy = deepcopy(critic)
            critic_copy = critic_copy.to(self.device)
            self.target_networks.append(critic_copy)
        self.target_networks = nn.ModuleList(self.target_networks)

        # Save a copy of original actor
        self.actor = deepcopy(actor)
        for param in self.actor.parameters():
            param.requires_grad = False

        self.target_actor = deepcopy(actor)

        self.n_critics = n_critics

        self.exploration_noise = exploration_noise

    def get_random_indices(self, sz=None, num_ind=2):
        # get num_ind random indices from a set of size sz (used for getting critic targets)

        if sz is None:
            sz = self.n_critics

        perm = torch.randperm(sz)
        ind = perm[:num_ind].to(self.device)

        return ind

    def loss_critic(self, obs, next_obs, actions, rewards, dones, gamma):

        # get random critic index
        critic_ind = self.get_random_indices()
        q1_ind = critic_ind[0]
        q2_ind = critic_ind[1]

        with torch.no_grad():
            target_q1 = self.target_networks[q1_ind]
            target_q2 = self.target_networks[q2_ind]    


            # get rl action using learned policy
            action_rl = super().forward(
                cond=obs,
                deterministic=False,
                network_override=self.target_actor,
            )

            # get il action using frozen policy
            action_il = super().forward(
                cond=obs,
                deterministic=False,
                network_override=self.actor,
            )

            # get random critic index
            critic_ind = self.get_random_indices()
            q1_ind = critic_ind[0]
            q2_ind = critic_ind[1]

            # get q values for both policies 
            q_rl1 = self.target_networks[q1_ind](obs, action_rl)[0]
            q_rl2 = self.target_networks[q2_ind](obs, action_rl)[0]
            q_rl = torch.min(q_rl1, q_rl2)

            q_il1 = self.target_networks[q1_ind](obs, action_il)[0]
            q_il2 = self.target_networks[q2_ind](obs, action_il)[0]
            q_il = torch.min(q_il1, q_il2)
            next_q = torch.max(q_rl, q_il)

            # terminal state mask
            mask = 1 - dones

            # flatten
            rewards = rewards.view(-1)
            next_q = next_q.view(-1)
            mask = mask.view(-1)

            # target value
            target_q = rewards + gamma * next_q * mask  # (B,)

        # loop over all critic networks and compute value estimate
        current_q = []
        for i in range(self.n_critics):
            current_q_i = self.critic_networks[i](obs, actions)[0]
            current_q.append(current_q_i)
        current_q = torch.stack(current_q, dim=-1)  # (B, n_critics)
        loss_critic = torch.mean((current_q - target_q.unsqueeze(-1)) ** 2)
        return loss_critic
    
    def loss_actor(self, obs):

        # compute current action and entropy
        action = self.forward(obs, deterministic=False)

        # loop over all critic networks and compute value estimate
        current_q = []
        for i in range(self.n_critics):
            current_q_i = self.critic_networks[i](obs, action)[0] 
            current_q.append(current_q_i)
        current_q = torch.stack(current_q, dim=-1) # (B, n_critics)
        current_q = torch.min(current_q, dim=-1).values
        
        loss_actor = -torch.mean(current_q)

        return loss_actor
    

    def update_target_critic(self, rho):
        soft_update(self.target_networks, self.critic_networks, rho)

    def update_target_actor(self, rho):
        soft_update(self.target_actor, self.network, rho)

    # ---------- Sampling ----------#

    def forward(
        self,
        cond,
        deterministic=False,
        use_base_policy=False,
    ):
        # get rl action using learned policy
        action_rl = super().forward(
            cond=cond,
            deterministic=deterministic,
            network_override=None,
        )
        action_rl = action_rl + torch.randn_like(action_rl) * self.exploration_noise

        # get il action using frozen policy
        action_il = super().forward(
            cond=cond,
            deterministic=deterministic,
            network_override=self.actor,
        )

        # get random critic index
        critic_ind = self.get_random_indices()
        q1_ind = critic_ind[0]
        q2_ind = critic_ind[1]

        # get q values for both policies 
        q_rl1 = self.target_networks[q1_ind](cond, action_rl)[0]
        q_rl2 = self.target_networks[q2_ind](cond, action_rl)[0]
        q_rl = torch.min(q_rl1, q_rl2)

        q_il1 = self.target_networks[q1_ind](cond, action_il)[0]
        q_il2 = self.target_networks[q2_ind](cond, action_il)[0]
        q_il = torch.min(q_il1, q_il2)

        # take action with higher q value
        action = torch.zeros_like(action_rl)
        action[q_rl > q_il] = action_rl[q_rl > q_il]
        action[q_rl <= q_il] = action_il[q_rl <= q_il]

        return action

        