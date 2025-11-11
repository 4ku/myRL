import random
import jax
import numpy as np
import optax
from flax.training import checkpoints
import argparse
import os

from configs.cartpole_dqn import Config
from jaxrl.agents.base_model import BaseModel

def evaluate_agent(agent, env, num_episodes=5, seed=1):
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
    return total_reward

def main(args):
    config = Config()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = jax.random.key(args.seed)

    # Create environment
    env = config.get_environment()

    # Create agent
    agent: BaseModel = config.agent.agent.create(
        rng=rng,
        observation_sample=env.observation_space.sample(),
        action_sample=env.action_space.sample(),
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
    for i in range(args.eval_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            # action = agent.sample_actions(obs)
            # action = jax.device_get(action)
            action = env.action_space.sample()
            next_obs, reward, done, truncated, infos = env.step(action)
            total_reward += reward
            obs = next_obs
        print(f"Episode {i+1}: Total reward = {total_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="experiments/CartPole-v1/DQN/20251111_20:04:22/checkpoints/",
    )
    parser.add_argument("--checkpoint_step", type=int, default=100_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args)
