from dataclasses import dataclass
import flax.linen as nn
import gymnasium as gym
import optax
import jax

from jaxrl.networks.mlp import MLP
from jaxrl.agents.discrete.vpg import VPG


@dataclass
class Config:
    seed: int = 1
    discount_factor: float = 0.99
    total_timesteps: int = 1_000_001
    batch_size: int = 128
    checkpoint_period: int = 50_000
    on_policy: bool = True
    
    buffer_size: int = 1_000
    buffer_sequence_length: int = 1

    # Evaluation
    eval_every: int = 25_000
    eval_episodes: int = 5

    def get_environment(self) -> gym.Env:
        env = gym.make("CartPole-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def get_eval_environment(self, video_folder: str) -> gym.Env:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(
            env, video_folder, episode_trigger=lambda x: True
        )
        return env

    def get_agent(
        self, rng: jax.Array, observation_space: gym.Space, action_space: gym.Space) -> VPG:
        agent = VPG.create(
            rng=rng,
            observation_sample=observation_space.sample(),
            action_dim=action_space.n,
            optimizer=optax.chain(
                optax.clip_by_global_norm(10.0),
                optax.adam(learning_rate=5e-4),
            ),
            network=MLP(
                hidden_dims=(256,),
                activation=nn.swish,
                use_layer_norm=True,
                dropout_rate=0.03,
            ),
            gamma=self.discount_factor,
        )

        return agent
