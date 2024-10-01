# Diffusion Policy Policy Optimization (DPPO)

## Installation 

1. Install core dependencies with a conda environment (if you do not plan to use Furniture-Bench, a higher Python version such as 3.10 can be installed instead) on a Linux machine with a Nvidia GPU.
```console
conda create -n dppo python=3.8 -y
conda activate dppo
pip install -e .
```

2. Install specific environment dependencies (Gym / Robomimic) or all dependencies.
```console
pip install -e .[gym] # or [robomimic]
pip install -e .[all]
```

3. [Install MuJoCo for Gym and/or Robomimic](installation/install_mujoco.md).

4. Set environment variables for data and logging directory (default is `data/` and `log/`), and set WandB entity (username or team name)
```
source script/set_path.sh
```

## Usage - Pre-training

Pre-training data for all tasks are pre-processed and can be found at `data/`.

### Run pre-training with data
All the configs can be found under `cfg/<env>/pretrain/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
```console
# Gym - hopper/walker2d/halfcheetah
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/gym/pretrain/hopper-medium-v2
# Robomimic - lift/can/square/transport
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/robomimic/pretrain/can
# D3IL - avoid_m1/m2/m3
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/d3il/pretrain/avoid_m1
# Furniture-Bench - one_leg/lamp/round_table_low/med
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/furniture/pretrain/one_leg_low
```

## Usage - Fine-tuning

### Fine-tuning pre-trained policy

To fine-tune the pre-trained policy, override `base_policy_path` to the checkpoint, which is saved under `checkpoint/` of the pre-training directory. Set `base_policy_path=<path>` in the command line when launching fine-tuning.

All the configs can be found under `cfg/<env>/finetune/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
```console
# Gym - hopper/walker2d/halfcheetah
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/gym/finetune/hopper-v2
# Robomimic - lift/can/square/transport
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/robomimic/finetune/can
# D3IL - avoid_m1/m2/m3
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/d3il/finetune/avoid_m1
# Furniture-Bench - one_leg/lamp/round_table_low/med
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/furniture/finetune/one_leg_low
```

**Note**: In Gym, Robomimic, and D3IL tasks, we run 40, 50, and 50 parallelized MuJoCo environments on CPU, respectively. If you would like to use fewer environments (given limited CPU threads, or GPU memory for rendering), you can reduce `env.n_envs` and increase `train.n_steps`, so the total number of environment steps collected in each iteration (n_envs x n_steps x act_steps) remains roughly the same. Try to set `train.n_steps` a multiple of `env.max_episode_steps / act_steps`, and be aware that we only count episodes finished within an iteration for eval. Furniture-Bench tasks run IsaacGym on a single GPU.


## Usage - Evaluation
Pre-trained or fine-tuned policies can be evaluated without running the fine-tuning script now. Some example configs are provided under `cfg/{gym/robomimic/furniture}/eval}` including ones below. Set `base_policy_path` to override the default checkpoint. 
```console
python script/run.py --config-name=eval_diffusion_mlp \
    --config-dir=cfg/gym/eval/hopper-v2
python script/run.py --config-name=eval_{diffusion/gaussian}_mlp_{?img} \
    --config-dir=cfg/robomimic/eval/can
python script/run.py --config-name=eval_diffusion_mlp \
    --config-dir=cfg/furniture/eval/one_leg_low
```

## DPPO implementation

Our diffusion implementation is mostly based on [Diffuser](https://github.com/jannerm/diffuser) and at [`model/diffusion/diffusion.py`](model/diffusion/diffusion.py) and [`model/diffusion/diffusion_vpg.py`](model/diffusion/diffusion_vpg.py). PPO specifics are implemented at [`model/diffusion/diffusion_ppo.py`](model/diffusion/diffusion_ppo.py). The main training script is at [`agent/finetune/train_ppo_diffusion_agent.py`](agent/finetune/train_ppo_diffusion_agent.py) that follows [CleanRL](https://github.com/vwxyzjn/cleanrl).

### Key configurations
* `denoising_steps`: number of denoising steps (should always be the same for pre-training and fine-tuning regardless the fine-tuning scheme)
* `ft_denoising_steps`: number of fine-tuned denoising steps
* `horizon_steps`: predicted action chunk size (should be the same as `act_steps`, executed action chunk size, with MLP. Can be different with UNet, e.g., `horizon_steps=16` and `act_steps=8`)
* `model.gamma_denoising`: denoising discount factor
* `model.min_sampling_denoising_std`: <img src="https://latex.codecogs.com/gif.latex?\epsilon^\text{exp}_\text{min} "/>, minimum amount of noise when sampling at a denoising step
* `model.min_logprob_denoising_std`: <img src="https://latex.codecogs.com/gif.latex?\epsilon^\text{prob}_\text{min} "/>, minimum standard deviation when evaluating likelihood at a denoising step
* `model.clip_ploss_coef`: PPO clipping ratio

### DDIM fine-tuning

To use DDIM fine-tuning, set `denoising_steps=100` in pre-training and set `model.use_ddim=True`, `model.ddim_steps` to the desired number of total DDIM steps, and `ft_denoising_steps` to the desired number of fine-tuned DDIM steps. In our Furniture-Bench experiments we use `denoising_steps=100`, `model.ddim_steps=5`, and `ft_denoising_steps=5`.

## Adding your own dataset/environment

### Pre-training data
Pre-training script is at [`agent/pretrain/train_diffusion_agent.py`](agent/pretrain/train_diffusion_agent.py). The pre-training dataset [loader](agent/dataset/sequence.py) assumes a npz file containing numpy arrays `states`, `actions`, `images` (if using pixel; img_h = img_w and a multiple of 8) and `traj_lengths`, where `states` and `actions` have the shape of num_total_steps x obs_dim/act_dim, `images` num_total_steps x C (concatenated if multiple images) x H x W, and `traj_lengths` is a 1-D array for indexing across num_total_steps.

#### Observation history
In our experiments we did not use any observation from previous timesteps (state or pixel), but it is implemented. You can set `cond_steps=<num_state_obs_step>` (and `img_cond_steps=<num_img_obs_step>`, no larger than `cond_steps`) in pre-training, and set the same when fine-tuning the newly pre-trained policy.

### Fine-tuning environment
We follow the Gym format for interacting with the environments. The vectorized environments are initialized at [make_async](env/gym_utils/__init__.py#L10) (called in the parent fine-tuning agent class [here](agent/finetune/train_agent.py#L38-L39)). The current implementation is not the cleanest as we tried to make it compatible with Gym, Robomimic, Furniture-Bench, and D3IL environments, but it should be easy to modify and allow using other environments. We use [multi_step](env/gym_utils/wrapper/multi_step.py) wrapper for history observations and multi-environment-step action execution. We also use environment-specific wrappers such as [robomimic_lowdim](env/gym_utils/wrapper/robomimic_lowdim.py) and [furniture](env/gym_utils/wrapper/furniture.py) for observation/action normalization, etc. You can implement a new environment wrapper if needed.

## Known issues
* IsaacGym simulation can become unstable at times and lead to NaN observations in Furniture-Bench. The current env wrapper does not handle NaN observations.
