from dataclasses import dataclass, field
from flax.struct import PyTreeNode
import flax.linen as nn
import gymnasium as gym

from jaxrl.networks.mlp import MLP
from jaxrl.agents.dqn import DQN

@dataclass
class Agent:
    agent: PyTreeNode = DQN
    hyperparams: dict = field(default_factory=lambda: {
        "gamma": 0.99,
        "tau": 0.005,
    })
    network: nn.Module = MLP(
        hidden_dims=(128, 128),
        activation=nn.swish,
        use_layer_norm=True,
        dropout_rate=0.1,
    )
    learning_rate: float = 5e-4


@dataclass
class Config():
    agent: Agent = field(default_factory=Agent)
    seed: int = 1
    total_timesteps: int = 200_001
    buffer_size: int = 100_000
    batch_size: int = 256
    utd_ratio: int = 4
    checkpoint_period: int = 20_000
    # Exploration
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.5
    # Evaluation
    eval_every: int = 10_000
    eval_episodes: int = 5

    def get_environment(self) -> gym.Env:
        env = gym.make("CartPole-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def get_eval_environment(self, video_folder: str) -> gym.Env:
        env = gym.make("CartPole-v1", render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)
        return env