"""
Reinforcement learning with prior data (RLPD) for Diffusion Policy.

Use ensemble of critics.

"""

import torch
import torch.nn as nn
import logging
from copy import deepcopy

from model.diffusion.diffusion_rwr import RWRDiffusion
from model.diffusion.sampling import make_timesteps

log = logging.getLogger(__name__)


class RLPD_Diffusion(RWRDiffusion):
    def __init__(
        self,
        actor,
        critic,
        n_critics,
        backup_entropy=False,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.n_critics = n_critics
        self.backup_entropy = backup_entropy

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

        # Construct a "stateless" version of one of the models. It is "stateless" in the sense that the parameters are meta Tensors and do not have storage.
        base_model = deepcopy(self.critic_networks[0])
        self.base_model = base_model.to("meta")
        self.ensemble_params, self.ensemble_buffers = torch.func.stack_module_state(
            self.critic_networks
        )

    def critic_wrapper(self, params, buffers, data):
        """for vmap"""
        return torch.func.functional_call(self.base_model, (params, buffers), data)

    def get_random_indices(self, sz=None, num_ind=2):
        """get num_ind random indices from a set of size sz (used for getting critic targets)"""
        if sz is None:
            sz = len(self.critic_networks)
        perm = torch.randperm(sz)
        ind = perm[:num_ind].to(self.device)
        return ind

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        rewards,
        terminated,
        gamma,
    ):
        # get random critic index
        q1_ind, q2_ind = self.get_random_indices()
        with torch.no_grad():
            next_actions = self.forward(
                cond=next_obs,
                deterministic=False,
            )
            next_q1 = self.target_networks[q1_ind](next_obs, next_actions)
            next_q2 = self.target_networks[q2_ind](next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2)

            # target value
            target_q = rewards + gamma * (1 - terminated) * next_q  # (B,)

        # run all critics in batch
        current_q = torch.vmap(self.critic_wrapper, in_dims=(0, 0, None))(
            self.ensemble_params, self.ensemble_buffers, (obs, actions)
        )  # (n_critics, B)
        loss_critic = torch.mean((current_q - target_q[None]) ** 2)
        return loss_critic

    def loss_actor(self, obs):
        action = self.forward_train(
            obs,
            deterministic=False,
        )
        current_q = torch.vmap(self.critic_wrapper, in_dims=(0, 0, None))(
            self.ensemble_params, self.ensemble_buffers, (obs, action)
        )  # (n_critics, B)
        current_q = current_q.mean(dim=0)
        loss_actor = -torch.mean(current_q)
        return loss_actor

    def update_target_critic(self, tau):
        """need to use ensemble_params instead of critic_networks"""
        for target_ind, target_critic in enumerate(self.target_networks):
            for target_param_name, target_param in target_critic.named_parameters():
                source_param = self.ensemble_params[target_param_name][target_ind]
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + source_param.data * tau
                )

    def forward_train(
        self,
        cond,
        deterministic=False,
    ):
        """
        Differentiable forward pass used in actor training.
        """
        device = self.betas.device
        B = len(cond["state"])

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
            )
            std = torch.exp(0.5 * logvar)

            # Determine the noise level
            if deterministic and t == 0:
                std = torch.zeros_like(std)
            elif deterministic:  # For DDPM, sample with noise
                std = torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=self.min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value and i == len(t_all) - 1:
                x = torch.clamp(x, -1, 1)
        return x
