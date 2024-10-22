"""
DPPO fine-tuning.

"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math

from collections import deque

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_sac_agent import TrainSACAgent
from util.scheduler import CosineAnnealingWarmupRestarts


class TrainSACDiffusionAgent(TrainSACAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Burn-in period for critic
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # LR schedulers
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)

        # Eta - between DDIM (=0 for eval) and DDPM (=1 for training)
        self.learn_eta = self.model.learn_eta
        if self.learn_eta:
            self.eta_update_interval = cfg.train.eta_update_interval
            self.eta_optimizer = torch.optim.AdamW(
                self.model.eta.parameters(),
                lr=cfg.train.eta_lr,
                weight_decay=cfg.train.eta_weight_decay,
            )
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.eta_optimizer,
                first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.eta_lr,
                min_lr=cfg.train.eta_lr_scheduler.min_lr,
                warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
                gamma=1.0,
            )

        # Scaling reward with constant
        self.reward_scale_const: float = cfg.train.get("reward_scale_const", 1)

    def run(self):
        # make a FIFO replay buffer for obs, action, and reward
        obs_buffer = deque(maxlen=self.buffer_size)
        next_obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        terminated_buffer = deque(maxlen=self.buffer_size)

        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:
            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) at the beginning
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or self.itr == 0:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = (
                        samples.trajectories.cpu().numpy()
                    )  # n_env x horizon x act
                    chains_venv = (
                        samples.chains.cpu().numpy()
                    )  # n_env x denoising x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv

                # add to buffer in train mode
                if not eval_mode:
                    for i in range(self.n_envs):
                        obs_buffer.append(prev_obs_venv["state"][i])
                        if "final_obs" in info_venv[i]:  # truncated
                            next_obs_buffer.append(info_venv[i]["final_obs"]["state"])
                        else:  # first obs in new episode
                            next_obs_buffer.append(obs_venv["state"][i])
                        action_buffer.append(chains_venv[i])
                    reward_buffer.extend(
                        (reward_venv * self.scale_reward_factor).tolist()
                    )
                    terminated_buffer.extend(terminated_venv.tolist())
                firsts_trajs[step + 1] = done_venv

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                if (
                    self.furniture_sparse_reward
                ):  # only for furniture tasks, where reward only occurs in one env step
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in reward_trajs_split
                        ]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            if not eval_mode and self.itr > self.n_explore_steps:
                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                total_env_steps = total_steps // self.model.ft_denoising_steps
                num_batch = max(1, total_steps // self.batch_size)  # skip last ones

                # Get the denoising indices (one per buffer index), also without replacement
                batch_denoising_inds = np.random.choice(
                    self.model.ft_denoising_steps * len(obs_buffer),
                    total_steps,
                    replace=False,
                )
                batch_denoising_inds = torch.Tensor(batch_denoising_inds).long()

                batch_inds, denoising_inds_b = torch.unravel_index(
                    batch_denoising_inds,
                    (len(obs_buffer), self.model.ft_denoising_steps),
                )

                for batch in range(num_batch):
                    start = batch * self.batch_size
                    end = start + self.batch_size
                    inds_b = batch_inds[start:end]
                    denoising_inds_b = denoising_inds_b[start:end]
                    obs_b = (
                        torch.from_numpy(np.array([obs_buffer[i] for i in inds_b]))
                        .float()
                        .to(self.device)
                    )
                    next_obs_b = (
                        torch.from_numpy(np.array([next_obs_buffer[i] for i in inds_b]))
                        .float()
                        .to(self.device)
                    )
                    # all next_obs_b where denoising_inds_b is less than self.model.ft_denoising_steps - 1 should be the same as obs_b
                    next_obs_b[
                        denoising_inds_b < self.model.ft_denoising_steps - 1
                    ] = obs_b[denoising_inds_b < self.model.ft_denoising_steps - 1]

                    chains_b = (
                        torch.from_numpy(
                            np.array(
                                [
                                    action_buffer[i][k]
                                    for i, k in zip(inds_b, denoising_inds_b)
                                ]
                            )
                        )
                        .float()
                        .to(self.device)
                    )
                    next_chains_b = (
                        torch.from_numpy(
                            np.array(
                                [
                                    action_buffer[i][k + 1]
                                    for i, k in zip(inds_b, denoising_inds_b)
                                ]
                            )
                        )
                        .float()
                        .to(self.device)
                    )
                    rewards_b = (
                        torch.from_numpy(np.array([reward_buffer[i] for i in inds_b]))
                        .float()
                        .to(self.device)
                    )
                    terminated_b = (
                        torch.from_numpy(
                            np.array([terminated_buffer[i] for i in inds_b])
                        )
                        .float()
                        .to(self.device)
                    )
                    obs_b = {"state": obs_b}
                    next_obs_b = {"state": next_obs_b}

                    # Update critic
                    alpha = self.log_alpha.exp().item()
                    loss_critic = self.model.loss_critic(
                        obs_b,
                        next_obs_b,
                        chains_b,
                        next_chains_b,
                        rewards_b,
                        terminated_b,
                        self.gamma,
                        alpha,
                    )
                    self.critic_optimizer.zero_grad()
                    loss_critic.backward()
                    self.critic_optimizer.step()

                    # Update target critic every critic update
                    self.model.update_target_critic(self.target_ema_rate)

                    # Delay update actor
                    loss_actor = 0
                    if self.itr % self.actor_update_freq == 0:
                        for _ in range(2):
                            loss_actor = self.model.loss_actor(
                                {"state": obs_b},
                                alpha,
                            )
                            self.actor_optimizer.zero_grad()
                            loss_actor.backward()
                            self.actor_optimizer.step()

                            # Update temperature parameter
                            self.log_alpha_optimizer.zero_grad()
                            loss_alpha = self.model.loss_temperature(
                                {"state": obs_b},
                                self.log_alpha.exp(),  # with grad
                                self.target_entropy,
                            )
                            loss_alpha.backward()
                            self.log_alpha_optimizer.step()

                # extract scalars
                loss_actor = loss_actor.item()
                loss_critic = loss_critic.item()
                loss_entropy = loss_alpha.item()

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | loss critic {loss_critic:8.4f} | reward {avg_episode_reward:8.4f} | alpha {alpha:8.4f} | t {time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "actor loss": loss_actor,
                                "critic loss": loss_critic,
                                "entropy loss": loss_entropy,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0][
                                    "lr"
                                ],
                                "entropy lr": self.log_alpha_optimizer.param_groups[0][
                                    "lr"
                                ],
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
