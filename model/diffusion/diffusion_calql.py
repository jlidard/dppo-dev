"""
Calibrated Conservative Q-Learning (CalQL) for Gaussian policy.

"""

import torch
import torch.nn as nn
import logging
from copy import deepcopy
import numpy as np
import einops

from model.diffusion.diffusion_rwr import RWRDiffusion
from model.diffusion.sampling import make_timesteps

log = logging.getLogger(__name__)


class CalQL_Diffusion(RWRDiffusion):
    def __init__(
        self,
        actor,
        critic,
        network_path=None,
        cql_clip_diff_min=-np.inf,
        cql_clip_diff_max=np.inf,
        cql_min_q_weight=5.0,
        cql_n_actions=10,
        **kwargs,
    ):
        super().__init__(network=actor, network_path=None, **kwargs)
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_n_actions = cql_n_actions

        # initialize critic networks
        self.critic = critic.to(self.device)
        self.target_critic = deepcopy(critic).to(self.device)

        # Load pre-trained checkpoint - note we are also loading the pre-trained critic here
        if network_path is not None:
            checkpoint = torch.load(
                network_path,
                map_location=self.device,
                weights_only=True,
            )
            self.load_state_dict(
                checkpoint["model"],
                strict=True,
            )
            log.info("Loaded actor from %s", network_path)
        log.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        random_actions,
        rewards,
        returns,
        terminated,
        gamma,
    ):
        B = len(actions)

        # Get initial TD loss
        q_data1, q_data2 = self.critic(obs, actions)
        with torch.no_grad():
            # repeat for action samples
            next_obs_repeated = {
                "state": next_obs["state"].repeat_interleave(self.cql_n_actions, dim=0)
            }

            # Get the next actions
            next_actions = self.forward(
                next_obs_repeated,
                deterministic=False,
            )
            next_q1, next_q2 = self.target_critic(next_obs_repeated, next_actions)
            next_q = torch.min(next_q1, next_q2)

            # Reshape the next_q to match the number of samples
            next_q = next_q.view(B, self.cql_n_actions)  # (B, n_sample)

            # Get the max indices over the samples, and index into the next_q
            max_idx = torch.argmax(next_q, dim=1)
            next_q = next_q[torch.arange(B), max_idx]

            # Get the target Q values
            target_q = rewards + gamma * (1 - terminated) * next_q

        # TD loss
        td_loss_1 = nn.functional.mse_loss(q_data1, target_q)
        td_loss_2 = nn.functional.mse_loss(q_data2, target_q)

        # Get actions
        pi_actions = self.forward(
            obs,
            deterministic=False,
        )  # no gradient
        pi_next_actions = self.forward(
            next_obs,
            deterministic=False,
        )  # no gradient

        # Random action Q values
        n_random_actions = random_actions.shape[1]
        obs_sample_state = {
            "state": obs["state"].repeat_interleave(n_random_actions, dim=0)
        }
        random_actions = einops.rearrange(random_actions, "B N H A -> (B N) H A")

        # Get the random action Q-values
        q_rand_1, q_rand_2 = self.critic(obs_sample_state, random_actions)
        q_rand_1 = q_rand_1
        q_rand_2 = q_rand_2

        # Reshape the random action Q values to match the number of samples
        q_rand_1 = q_rand_1.view(B, n_random_actions)  # (n_sample, B)
        q_rand_2 = q_rand_2.view(B, n_random_actions)

        # Policy action Q values
        q_pi_1, q_pi_2 = self.critic(obs, pi_actions)
        q_pi_next_1, q_pi_next_2 = self.critic(next_obs, pi_next_actions)

        # Ensure calibration w.r.t. value function estimate
        q_pi_1 = torch.max(q_pi_1, returns)[:, None]  # (B, 1)
        q_pi_2 = torch.max(q_pi_2, returns)[:, None]  # (B, 1)
        q_pi_next_1 = torch.max(q_pi_next_1, returns)[:, None]  # (B, 1)
        q_pi_next_2 = torch.max(q_pi_next_2, returns)[:, None]  # (B, 1)

        # cql_importance_sample
        q_pi_1 = q_pi_1
        q_pi_2 = q_pi_2
        q_pi_next_1 = q_pi_next_1
        q_pi_next_2 = q_pi_next_2
        cat_q_1 = torch.cat(
            [q_rand_1, q_pi_1, q_pi_next_1], dim=-1
        )  # (B, num_samples+1)
        cql_qf1_ood = torch.logsumexp(cat_q_1, dim=-1)  # max over num_samples
        cat_q_2 = torch.cat(
            [q_rand_2, q_pi_2, q_pi_next_2], dim=-1
        )  # (B, num_samples+1)
        cql_qf2_ood = torch.logsumexp(cat_q_2, dim=-1)  # sum over num_samples

        # skip cal_lagrange since the paper shows cql_target_action_gap not used in kitchen

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

    def loss_actor(self, obs):
        action = self.forward_train(
            obs,
            deterministic=False,
        )
        q1, q2 = self.critic(obs, action)
        actor_loss = -torch.min(q1, q2)
        return actor_loss.mean()

    def update_target_critic(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

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
