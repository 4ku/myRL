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
    if not config.on_policy:
        raise ValueError("Use train_off_policy.py for off-policy configs.")

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
        "log_prob": 0.0,
        "return": 0.0,
    }
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
    sequence_length = config.buffer_sequence_length

    for global_step in tqdm(range(config.total_timesteps), desc="Training"):
        # Sample action
        rng, _ = jax.random.split(rng)
        action, action_log_prob = agent.sample_actions(obs, rng, argmax=False)
        action = jax.device_get(action)
        action_log_prob = jax.device_get(action_log_prob)

        # Execute environment step
        next_obs, reward, done, truncated, infos = env.step(action)

        transition = {
            "observation": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "log_prob": action_log_prob,
        }
        episode_transitions.append(transition)
        obs = next_obs

        if done or truncated:
            episode_transitions = compute_returns(
                episode_transitions, config.discount_factor
            )
            for t in episode_transitions:
                replay_buffer.insert(t)
            episode_transitions = []
            log_info(writer, infos, global_step)
            seed = np.random.randint(0, 2**31 - 1)
            obs, _ = env.reset(seed=seed)

        # Update agent
        required = config.batch_size + sequence_length - 1
        if replay_buffer.size >= required:
            agent_info = {}
            while replay_buffer.size >= required:
                batch = replay_buffer.sample(
                    batch_size=config.batch_size,
                    sequence_length=sequence_length,
                    pop=True,
                )
                batch = jax.device_put(batch)
                agent, agent_info = agent.update(batch, rng)
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
            eval_infos = evaluate(
                eval_env, agent, num_episodes=config.eval_episodes, seed=config.seed
            )
            log_info(writer, eval_infos, global_step)

    # Wait for any pending checkpoint saves to complete
    checkpointer.wait_until_finished()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent (on-policy)")
    parser.add_argument(
        "--config",
        type=str,
        default="pendulum_vpg",
        help="Config module name from configs folder (e.g., cartpole_vpg)",
    )
    args = parser.parse_args()
    main(args)