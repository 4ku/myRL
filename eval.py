import argparse
import os
import importlib
import random
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp

from jaxrl.agents.base_model import BaseModel

def evaluate(env, agent, num_episodes=5, seed=1):
    for i in range(num_episodes):
        obs, info = env.reset(seed=seed + i)
        done = False
        truncated = False
        total_reward = 0
        while not done and not truncated:
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
    video_path = os.path.dirname(os.path.normpath(args.checkpoint_path))
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
    checkpoint_path = os.path.abspath(f"{args.checkpoint_path}/{args.checkpoint_step}")
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = jax.tree.map(
        lambda _: ocp.ArrayRestoreArgs(sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])),
        agent.state,
    )
    restored_state = checkpointer.restore(checkpoint_path, item=agent.state, restore_args=restore_args)
    agent = agent.replace(state=restored_state)

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
        default="experiments/CartPole-v1/DQN/20251125_10:57:30/checkpoints",
    )
    parser.add_argument("--checkpoint_step", type=int, default=100_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args)
