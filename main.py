import random
import time
import importlib

import gymnasium as gym
import jax
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer

# Import our universal config system
from configs.config import get_config, get_predefined_config, ALGORITHMS, PREDEFINED_CONFIGS


def load_agent_class(module_path: str, class_name: str):
    """Dynamically load an agent class from a module path."""
    try:
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        return agent_class
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {e}")


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def record_video(agent, env_id, video_folder, episodes=5, seed=1):
    """Record video of agent performance."""
    env_vid = gym.make(env_id, render_mode="rgb_array")
    env_vid = gym.wrappers.RecordVideo(
        env_vid, video_folder=video_folder, episode_trigger=lambda _: True
    )
    rewards = []
    for ep in range(episodes):
        reset_out = env_vid.reset(seed=seed)
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action = agent.sample_actions(obs)
            action = jax.device_get(action)
            step_out = env_vid.step(action)
            if len(step_out) == 5:
                obs, r, term, trunc, _ = step_out
                done = term
                truncated = trunc
            else:
                obs, r, done, _ = step_out
            ep_reward += r
        rewards.append(ep_reward)
    env_vid.close()
    avg_reward = np.mean(rewards)
    return avg_reward


def evaluate_agent(agent, env_id, n_episodes=5, seed=1, deterministic=True):
    """Evaluate agent performance without recording video."""
    env_eval = gym.make(env_id)
    rewards = []
    lengths = []
    
    for ep in range(n_episodes):
        reset_out = env_eval.reset(seed=seed + ep)  # Different seed for each episode
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out
        done = False
        truncated = False
        ep_reward = 0.0
        ep_length = 0
        
        while not (done or truncated):
            if deterministic:
                # Use deterministic action selection for evaluation
                action = agent.sample_actions(obs)
                action = jax.device_get(action)
            else:
                # Use stochastic action selection
                action = agent.sample_actions(obs)
                action = jax.device_get(action)
            
            step_out = env_eval.step(action)
            if len(step_out) == 5:
                obs, r, term, trunc, _ = step_out
                done = term
                truncated = trunc
            else:
                obs, r, done, _ = step_out
            ep_reward += r
            ep_length += 1
            
        rewards.append(ep_reward)
        lengths.append(ep_length)
    
    env_eval.close()
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'rewards': rewards,
        'lengths': lengths
    }


def main(config_name: str = None, algorithm: str = "dqn", **config_overrides):
    """
    Run a reinforcement learning experiment.
    
    Args:
        config_name: Name of predefined config to use (e.g., "dqn_cartpole")
        algorithm: Algorithm to use if config_name is not provided
        **config_overrides: Any configuration parameters to override
    """
    # Load configuration
    if config_name:
        if config_name not in PREDEFINED_CONFIGS:
            print(f"Available predefined configs: {list(PREDEFINED_CONFIGS.keys())}")
            raise ValueError(f"Config '{config_name}' not found")
        config = get_predefined_config(config_name)
    else:
        if algorithm not in ALGORITHMS:
            print(f"Available algorithms: {list(ALGORITHMS.keys())}")
            raise ValueError(f"Algorithm '{algorithm}' not found")
        config = get_config(algorithm)
    
    # Apply any overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.agent, key):
            setattr(config.agent, key, value)
        elif key in config.agent.hyperparameters:
            config.agent.hyperparameters[key] = value
        else:
            config.agent.hyperparameters[key] = value
    
    # Create run name
    run_name = f"{config.env_id}__{config.agent.name}__{config.exp_name}__{config.seed}__{int(time.time())}"
    
    print(f"Running experiment: {run_name}")
    print(f"Algorithm: {config.agent.name}")
    print(f"Environment: {config.env_id}")
    print(f"Hyperparameters: {config.agent.hyperparameters}")
    
    # Setup tracking
    if config.track:
        import wandb
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config={
                **vars(config),
                **config.agent.hyperparameters,
                "algorithm": config.agent.name,
            },
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in {
            **vars(config),
            **config.agent.hyperparameters,
        }.items()])),
    )

    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.env_id, config.seed, 0, config.capture_video, run_name)],
        autoreset_mode="SameStep",
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Dynamically load and create agent
    agent_class = load_agent_class(config.agent.module_path, config.agent.class_name)
    agent = agent_class.create(
        rng=key,
        observation=envs.single_observation_space.sample(),
        action_dim=envs.single_action_space.n,
        optimizer=optax.adam(learning_rate=config.learning_rate),
        **config.agent.hyperparameters,
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        config.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    start_time = None

    # Training loop
    obs, _ = envs.reset(seed=config.seed)
    for global_step in range(config.total_timesteps):
        # Epsilon-greedy exploration (if applicable)
        epsilon = linear_schedule(
            config.start_e,
            config.end_e,
            config.exploration_fraction * config.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = agent.sample_actions(obs)
            actions = jax.device_get(actions)

        # Execute environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log episode returns
        if "_final_info" in infos:
            finfo = infos["final_info"]
            mask = infos["_final_info"]
            returns = finfo["episode"]["r"]
            lengths = finfo["episode"]["l"]
            for idx, done_flag in enumerate(mask):
                if done_flag:
                    r = returns[idx]
                    l = lengths[idx]
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)
                    print(f"Episode {global_step} reward: {r}, length: {l}")

        # Store transition in replay buffer
        real_next_obs = next_obs.copy()
        if "final_obs" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_obs"][idx]
        replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Training
        if global_step > config.learning_starts:
            if start_time is None:
                start_time = time.time()
            
            if global_step % config.train_frequency == 0:
                data = replay_buffer.sample(config.batch_size)
                data = {
                    "observations": data.observations.numpy(),
                    "actions": data.actions.flatten().numpy(),
                    "next_observations": data.next_observations.numpy(),
                    "rewards": data.rewards.flatten().numpy(),
                    "dones": data.dones.flatten().numpy(),
                }
                agent, info = agent.update(data)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", jax.device_get(info["loss"]), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(info["q_pred"]).mean(), global_step)
                    writer.add_scalar(
                        "charts/SPS",
                        int((global_step - config.learning_starts) / (time.time() - start_time)),
                        global_step,
                    )
                
                if global_step % 1000 == 0 and global_step > 0:
                    print(f"Global step: {global_step}")
                    print(f"Loss: {jax.device_get(info['loss'])}")
                    print(f"Q-values: {jax.device_get(info['q_pred']).mean()}")
                    print(f"SPS: {int((global_step - config.learning_starts) / (time.time() - start_time))}")
                    print("--------------------------------")

        # Periodic evaluation
        if (global_step > config.learning_starts and 
            global_step % config.eval_every == 0 and 
            global_step > 0):
            
            eval_results = evaluate_agent(
                agent, 
                config.env_id, 
                n_episodes=config.eval_episodes, 
                seed=config.seed + 1000,  # Different seed for evaluation
                deterministic=config.eval_deterministic
            )
            
            # Log evaluation results
            writer.add_scalar("eval/mean_reward", eval_results['mean_reward'], global_step)
            writer.add_scalar("eval/std_reward", eval_results['std_reward'], global_step)
            writer.add_scalar("eval/mean_length", eval_results['mean_length'], global_step)
            
            print(f"Evaluation at step {global_step}:")
            print(f"  Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"  Mean length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
            print("--------------------------------")

    envs.close()
    writer.close()

    # Final evaluation
    if config.capture_video:
        avg_r = record_video(
            agent,
            config.env_id,
            video_folder=f"videos/step_{config.total_timesteps}",
            episodes=5,
            seed=config.seed,
        )
        print(f"Average reward: {avg_r:.2f}")
    else:
        # Quick evaluation without video
        env_eval = gym.make(config.env_id)
        rewards = []
        for ep in range(5):
            reset_out = env_eval.reset(seed=config.seed)
            if isinstance(reset_out, tuple):
                obs, _ = reset_out
            else:
                obs = reset_out
            done = False
            truncated = False
            ep_reward = 0.0
            while not (done or truncated):
                action = agent.sample_actions(obs)
                action = jax.device_get(action)
                step_out = env_eval.step(action)
                if len(step_out) == 5:
                    obs, r, term, trunc, _ = step_out
                    done = term
                    truncated = trunc
                else:
                    obs, r, done, _ = step_out
                ep_reward += r
            rewards.append(ep_reward)
        env_eval.close()
        avg_r = np.mean(rewards)
        print(f"Average reward: {avg_r:.2f}")
    
    return agent, avg_r


if __name__ == "__main__":
    main()
        