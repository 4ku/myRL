from dataclasses import dataclass
import flax.linen as nn
import gymnasium as gym
import optax
import jax

from jaxrl.networks.mlp import MLP
from jaxrl.agents.continuous.td3 import TD3


@dataclass
class Config:
    seed: int = 1
    discount_factor: float = 0.99
    total_timesteps: int = 100_000
    buffer_size: int = 100_000
    batch_size: int = 256
    utd_ratio: int = 1
    checkpoint_period: int = 10_000
    on_policy: bool = False
    
    # Exploration
    # Using small epsilon to allow some random exploration, 
    # relying on deterministic policy otherwise.
    start_e: float = 0.2
    end_e: float = 0.0
    exploration_fraction: float = 0.1
    
    # Evaluation
    eval_every: int = 10_000
    eval_episodes: int = 5

    def get_environment(self) -> gym.Env:
        env = gym.make("Pendulum-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def get_eval_environment(self, video_folder: str) -> gym.Env:
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(
            env, video_folder, episode_trigger=lambda x: True
        )
        return env

    def get_agent(
        self, rng: jax.Array, observation_space: gym.Space, action_space: gym.Space) -> TD3:
        
        actor_net = MLP(
            hidden_dims=(256,),
            activation=nn.relu,
            use_layer_norm=False,
            dropout_rate=0.0,
        )
        
        critic_net = MLP(
            hidden_dims=(256,),
            activation=nn.relu,
            use_layer_norm=False,
            dropout_rate=0.0,
        )

        agent = TD3.create(
            rng=rng,
            observation_sample=observation_space.sample(),
            action_space=action_space,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=optax.adam(learning_rate=3e-4),
            critic_optimizer=optax.adam(learning_rate=3e-4),
            gamma=self.discount_factor,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            critic_ensemble_size=10,
            critic_subsample_size=2,
        )

        return agent
