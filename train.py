import argparse
import datetime
import importlib
import os
import random
import jax
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import orbax.checkpoint as ocp

from datastore.replay_buffer import ReplayBuffer
from jaxrl.agents.base_model import BaseModel
from utils import log_info, compute_returns
from eval import evaluate


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
    agent: BaseModel = config.get_agent(
        rng=rng,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    example_transition = {
        "observation": env.observation_space.sample(),
        "action": env.action_space.sample(),
        "reward": 0.0,
        "done": False,
        "truncated": False,
        "return": 0.0,
    }
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        example_transition=example_transition,
        capacity=config.buffer_size,
        seed=config.seed,
    )

    # Setup logging
    now = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    experiment_folder = (
        f"experiments/{env.unwrapped.spec.id}/{agent.__class__.__name__}/{now}"
    )
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    writer = SummaryWriter(f"{experiment_folder}/tensorboard")
    writer.add_text("config", str(config))

    # Setup checkpointing
    checkpoint_dir = os.path.abspath(f"{experiment_folder}/checkpoints")
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

    # Training loop
    obs, _ = env.reset(seed=config.seed)
    episode_transitions = []
    
    for global_step in tqdm(range(config.total_timesteps), desc="Training"):

        # Get epsilon from linear schedule
        slope = (config.end_e - config.start_e) / (
            config.exploration_fraction * config.total_timesteps
        )
        epsilon = max(slope * global_step + config.start_e, config.end_e)
        writer.add_scalar("charts/epsilon", epsilon, global_step)

        # Sample action
        rng, _ = jax.random.split(rng)
        action = (
            agent.sample_actions(obs, rng)
            if random.random() > epsilon
            else env.action_space.sample()
        )
        action = jax.device_get(action)

        # Execute environment step
        next_obs, reward, done, truncated, infos = env.step(action)

        transition = {
            "observation": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "truncated": truncated,
        }
        episode_transitions.append(transition)
        obs = next_obs

        if done or truncated:
            episode_transitions = compute_returns(episode_transitions, config.discount_factor)
            for transition in episode_transitions:
                replay_buffer.insert(transition)
            episode_transitions = []
            log_info(writer, infos, global_step)
            seed = np.random.randint(0, 2**31 - 1)
            obs, _ = env.reset(seed=seed)

        # Update agent
        if replay_buffer.size < config.batch_size:
            continue
        for _ in range(config.utd_ratio):
            data = replay_buffer.sample(config.batch_size)
            agent, agent_info = agent.update(data, rng)
            rng, _ = jax.random.split(rng)

        log_info(writer, agent_info, global_step)

        # Save checkpoint
        if global_step % config.checkpoint_period == 0:
            checkpointer.save(
                f"{checkpoint_dir}/{global_step}",
                args=ocp.args.PyTreeSave(agent.state),
            )

        # Evaluate agent
        if global_step % config.eval_every == 0:
            video_folder = f"{experiment_folder}/eval_videos/{global_step}"
            eval_env = config.get_eval_environment(video_folder)
            eval_infos = evaluate(eval_env, agent, num_episodes=config.eval_episodes, seed=config.seed)
            log_info(writer, eval_infos, global_step)

    # Wait for any pending checkpoint saves to complete
    checkpointer.wait_until_finished()


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
