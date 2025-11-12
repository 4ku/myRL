import argparse
import datetime
import importlib
import os
import random
import jax
import numpy as np
import optax
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from flax.training import checkpoints

from datastore.replay_buffer import ReplayBuffer
from jaxrl.agents.base_model import BaseModel
from utils import log_info


def main(args):
    # Dynamically import config module
    config_module = importlib.import_module(f"configs.{args.config}")
    config = config_module.Config()

    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    rng = jax.random.key(config.seed)

    # Create environment
    env = config.get_environment()

    # Create agent
    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
    else:
        if len(env.action_space.shape) > 1:
            raise ValueError("Action space must be a 1D array")
        action_dim = env.action_space.shape[0]
    
    agent: BaseModel = config.agent.agent.create(
        rng=rng,
        observation_sample=env.observation_space.sample(),
        action_dim=action_dim,
        optimizer=optax.adam(learning_rate=config.agent.learning_rate),
        network=config.agent.network,
        **config.agent.hyperparams,
    )

    example_transition = {
        "observation": env.observation_space.sample(),
        "action": env.action_space.sample(),
        "reward": 0.0,
        "done": False,
        "truncated": False,
    }
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        example_transition=example_transition,
        capacity=config.buffer_size,
    )

    # Setup logging
    now = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    experiment_folder = (
        f"experiments/{env.unwrapped.spec.id}/{config.agent.agent.__name__}/{now}"
    )
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    writer = SummaryWriter(f"{experiment_folder}/tensorboard")
    writer.add_text("config", str(config))

    # Training loop
    seed = int(jax.random.randint(rng, (), 0, 2**31 - 1))
    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0

    for global_step in tqdm(range(config.total_timesteps), desc="Training"):

        # Get epsilon from linear schedule
        slope = (config.end_e - config.start_e) / (
            config.exploration_fraction * config.total_timesteps
        )
        epsilon = max(slope * global_step + config.start_e, config.end_e)
        writer.add_scalar("charts/epsilon", epsilon, global_step)

        # Sample action
        action = (
            agent.sample_actions(obs)
            if random.random() > epsilon
            else env.action_space.sample()
        )
        action = jax.device_get(action)

        # Execute environment step
        next_obs, reward, done, truncated, infos = env.step(action)
        episode_reward += reward

        transition = {
            "observation": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "truncated": truncated,
        }

        # Store transition in replay buffer
        replay_buffer.insert(transition)
        obs = next_obs

        if done or truncated:
            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            episode_reward = 0.0
            rng, _ = jax.random.split(rng)
            seed = int(jax.random.randint(rng, (), 0, 2**31 - 1))
            obs, info = env.reset(seed=seed)

        # Update agent
        if replay_buffer.size < config.batch_size:
            continue
        for i in range(config.utd_ratio):
            data = replay_buffer.sample(config.batch_size)
            agent, info = agent.update(data)

        log_info(writer, info, global_step)

        # Save checkpoint
        if global_step % config.checkpoint_period == 0:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoints.save_checkpoint(
                f"{current_dir}/{experiment_folder}/checkpoints/",
                agent.state,
                step=global_step,
                keep=20,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument(
        "--config",
        type=str,
        default="cartpole_dqn",
        help="Config module name from configs folder (e.g., cartpole_dqn)",
    )
    args = parser.parse_args()
    main(args)
