import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import jax
import numpy as np
import optax
import tyro
from torch.utils.tensorboard import SummaryWriter
# from jaxrl.agents.data.replay_buffer import ReplayBuffer
from jaxrl.agents.discrete.dqn import DQN
import yaml
from stable_baselines3.common.buffers import ReplayBuffer
import jax.numpy as jnp
from flax.core import FrozenDict

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 257
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""

    # Agent specific arguments
    agent_name: str = "dqn"
    """the name of the agent"""
    agent_config_path: str = "configs/agents/discrete/dqn/default.yaml"
    """the path to the agent config"""


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def record_video(agent, env_id, video_folder, episodes=5):
    # create environment for recording with RGB array rendering
    env_vid = gym.make(env_id, render_mode="rgb_array")
    env_vid = gym.wrappers.RecordVideo(
        env_vid, video_folder=video_folder, episode_trigger=lambda _: True
    )
    rewards = []
    for ep in range(episodes):
        # reset environment
        reset_out = env_vid.reset(seed=args.seed)
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            # select action using trained Q-network
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

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
        ],
        autoreset_mode="SameStep",
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    with open(args.agent_config_path, "r") as f:
        agent_config = yaml.safe_load(f)

    agent = DQN.create(
        rng=key,
        observation=envs.single_observation_space.sample(),
        action_dim=envs.single_action_space.n,
        optimizer=optax.adam(learning_rate=args.learning_rate),
        **agent_config,
    )

    # replay_buffer = ReplayBuffer(
    #     capacity=args.buffer_size,
    #     obs_shape=envs.single_observation_space.shape,
    # )
    replay_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    start_time = None

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = agent.sample_actions(obs)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "_final_info" in infos:
            finfo = infos["final_info"]
            mask = infos["_final_info"]
            returns = finfo["episode"]["r"]
            lengths = finfo["episode"]["l"]
            for idx, done_flag in enumerate(mask):
                if done_flag:
                    r = returns[idx]
                    l = lengths[idx]
                    # print(f"global_step={global_step}, episodic_return={r:.2f}")
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_obs`
        real_next_obs = next_obs.copy()
        if "final_obs" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_obs"][idx]
        # replay_buffer.add(obs, actions, rewards, real_next_obs, terminations)
        replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if start_time is None:
                start_time = time.time()
            # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            if global_step % args.train_frequency == 0:
                # key, subkey = jax.random.split(key)
                # data = replay_buffer.sample(args.batch_size, key)
                data = replay_buffer.sample(args.batch_size)
                # data = FrozenDict({
                #     "observations": jnp.array(data.observations),
                #     "actions": jnp.array(data.actions),
                #     "next_observations": jnp.array(data.next_observations),
                #     "rewards": jnp.array(data.rewards),
                #     "dones": jnp.array(data.dones),
                # })
                data = {
                    "observations": data.observations.numpy(),
                    "actions": data.actions.flatten().numpy(),
                    "next_observations": data.next_observations.numpy(),
                    "rewards": data.rewards.flatten().numpy(),
                    "dones": data.dones.flatten().numpy(),
                }
                agent, info = agent.update(data)

                if global_step % 100 == 0:
                    writer.add_scalar(
                        "losses/td_loss", jax.device_get(info["loss"]), global_step
                    )
                    writer.add_scalar(
                        "losses/q_values", jax.device_get(info["q_pred"]).mean(), global_step
                    )
                    writer.add_scalar(
                        "charts/SPS",
                        int((global_step - args.learning_starts) / (time.time() - start_time)),
                        global_step,
                    )
                if global_step % 1000 == 0 and global_step > 0:
                    print("Global step:", global_step)
                    print("Loss:", jax.device_get(info["loss"]))
                    print("Q-values:", jax.device_get(info["q_pred"]).mean())
                    print("SPS:", int((global_step - args.learning_starts) / (time.time() - start_time)))
                    print("--------------------------------")
                # if global_step % 5000 == 0:
                #     avg_r = record_video(
                #         agent,
                #         args.env_id,
                #         video_folder=f"videos/step_{global_step}",
                #         episodes=5,
                #     )
                #     print(f"Average reward: {avg_r:.2f}")

    envs.close()
    writer.close()

    # Run evaluation and record video
    avg_r = record_video(
        agent,
        args.env_id,
        video_folder=f"videos/step_{args.total_timesteps}",
        episodes=5,
    )
    print(f"Average reward: {avg_r:.2f}")
