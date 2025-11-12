import argparse
import os
import importlib
import random
import jax
import numpy as np
import optax
from flax.training import checkpoints

from jaxrl.agents.base_model import BaseModel

def evaluate(env, agent, num_episodes=5, seed=1):
    for i in range(num_episodes):
        obs, info = env.reset(seed=seed + i)
        done = False
        total_reward = 0
        while not done:
            action = agent.sample_actions(obs)
            action = jax.device_get(action)
            next_obs, reward, done, truncated, infos = env.step(action)
            total_reward += reward
            obs = next_obs
        print(f"Episode {i+1}: Total reward = {total_reward}")
    env.close()

def main(args):
    # Dynamically import config module
    config_module = importlib.import_module(f"configs.{args.config}")
    config = config_module.Config()

    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    rng = jax.random.key(config.seed)

    # Create environment
    video_path = os.path.dirname(args.checkpoint_path)
    env = config.get_eval_environment(f"{video_path}/videos")

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
    
    # Load checkpoint
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint = checkpoints.restore_checkpoint(
        f"{current_dir}/{args.checkpoint_path}", agent.state, step=args.checkpoint_step
    )
    agent = agent.replace(state=checkpoint)

    # Evaluate agent
    evaluate(env, agent, num_episodes=args.eval_episodes, seed=args.seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="cartpole_dqn",
        help="Config module name from configs folder (e.g., cartpole_dqn)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="experiments/CartPole-v1/DQN/20251112_18:39:44/checkpoints",
    )
    parser.add_argument("--checkpoint_step", type=int, default=100_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args)
