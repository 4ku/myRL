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
    infos = []
    for i in range(num_episodes):
        obs, info = env.reset(seed=seed + i)
        rng = jax.random.key(seed + i)
        done = False
        truncated = False
        total_reward = 0
        while not done and not truncated:
            rng, action_rng = jax.random.split(rng)
            action = agent.sample_actions(obs, action_rng)
            action = jax.device_get(action)
            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            obs = next_obs
        infos.append(info)
        print(f"Episode {i+1}: Total reward = {total_reward}")
    env.close()

    return aggregate_infos(infos, prefix="eval")


def aggregate_infos(infos, prefix=""):
    """Recursively aggregate a list of info dicts by concatenating values for same keys."""
    aggregated = {}
    if not infos:
        return aggregated
    
    all_keys = set()
    for info in infos:
        all_keys.update(info.keys())
    
    for key in all_keys:
        values = [info[key] for info in infos if key in info]
        if not values:
            continue
        
        prefix_key = f"{prefix}/{key}" if prefix else key
        
        if isinstance(values[0], dict):
            # Recursively aggregate nested dicts
            nested = aggregate_infos(values, prefix=prefix_key)
            aggregated.update(nested)
        elif isinstance(values[0], (int, float, np.number, np.ndarray)):
            aggregated[prefix_key] = np.concatenate([np.atleast_1d(v) for v in values])
    
    return aggregated

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
    env = config.get_eval_environment(f"{video_path}/eval_videos/{args.checkpoint_step}")

    # Create agent
    agent = config.get_agent(
        rng=rng,
        observation_space=env.observation_space,
        action_space=env.action_space,
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
