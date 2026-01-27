from dataclasses import dataclass
import flax.linen as nn
import gymnasium as gym
import optax
import jax

from jaxrl.networks.mlp import MLP
from jaxrl.agents.continuous.sac import SAC


@dataclass
class Config:
    seed: int = 1
    discount_factor: float = 0.99
    total_timesteps: int = 150_000
    batch_size: int = 256
    utd_ratio: int = 1
    checkpoint_period: int = 100_000
    on_policy: bool = False
    
    buffer_size: int = 100_000
    buffer_sequence_length: int = 2

    # Exploration
    # SAC does exploration via entropy, but we might keep these for consistency 
    # though they are unused by SAC (it uses alpha).
    start_e: float = 0.0
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
        self, rng: jax.Array, observation_space: gym.Space, action_space: gym.Space) -> SAC:
        
        actor_net = MLP(
            hidden_dims=(256,),
            activation=nn.swish,
            use_layer_norm=True,
            dropout_rate=0.0,
        )
        
        critic_net = MLP(
            hidden_dims=(256,),
            activation=nn.swish,
            use_layer_norm=True,
            dropout_rate=0.0,
        )

        agent = SAC.create(
            rng=rng,
            observation_sample=observation_space.sample(),
            action_space=action_space,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=optax.adam(learning_rate=3e-4),
            critic_optimizer=optax.adam(learning_rate=3e-4),
            temperature_optimizer=optax.adam(learning_rate=3e-4),
            critic_ensemble_size=4,
            critic_subsample_size=2,
            actor_log_std_min=-20.0,
            actor_log_std_max=2.0,
            init_temperature=1.0,
            gamma=self.discount_factor,
            tau=0.005,
            target_entropy=-float(action_space.shape[-1]),
        )

        return agent
