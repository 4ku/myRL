from dataclasses import dataclass
import flax.linen as nn
import gymnasium as gym
import optax
import jax

from jaxrl.networks.mlp import MLP
from jaxrl.agents.discrete.dqn import DQN


@dataclass
class Config:
    seed: int = 1
    discount_factor: float = 0.99
    total_timesteps: int = 200_001
    buffer_size: int = 100_000
    batch_size: int = 512
    utd_ratio: int = 4
    checkpoint_period: int = 10_000
    on_policy: bool = False

    # Exploration
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    
    # Evaluation
    eval_every: int = 10_000
    eval_episodes: int = 5

    def get_environment(self) -> gym.Env:
        env = gym.make("MountainCar-v0")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        class RewardShapingWrapper(gym.Wrapper):
            def step(self, action):
                obs, reward, done, truncated, info = self.env.step(action)
                # Shaping: encourage velocity to build momentum
                reward += 10.0 * abs(obs[1])
                return obs, reward, done, truncated, info
        
        env = RewardShapingWrapper(env)
        return env

    def get_eval_environment(self, video_folder: str) -> gym.Env:
        env = gym.make("MountainCar-v0", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(
            env, video_folder, episode_trigger=lambda x: True
        )
        return env

    def get_agent(
        self, rng: jax.Array, observation_space: gym.Space, action_space: gym.Space) -> DQN:
        agent = DQN.create(
            rng=rng,
            observation_sample=observation_space.sample(),
            action_dim=action_space.n,
            optimizer=optax.chain(
                optax.clip_by_global_norm(50.0),
                optax.adam(learning_rate=5e-4),
            ),
            network=MLP(
                hidden_dims=(256,),
                activation=nn.swish,
                use_layer_norm=True,
                dropout_rate=0.03,
            ),
            gamma=self.discount_factor,
            tau=0.005,
            critic_ensemble_size=10,
            critic_subsample_size=2,
        )

        return agent
