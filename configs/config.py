from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class AgentConfig:
    """Base configuration for RL agents"""
    name: str
    module_path: str  # e.g., "jaxrl.agents.discrete.dqn"
    class_name: str   # e.g., "DQN"
    hyperparameters: Dict[str, Any]


@dataclass
class ExperimentConfig:
    """Configuration for the entire experiment"""
    # Experiment meta
    exp_name: str = "universal_rl"
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = True
    
    # Environment
    env_id: str = "CartPole-v1"
    total_timesteps: int = 100_000
    
    # Training
    learning_rate: float = 5e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    learning_starts: int = 257
    train_frequency: int = 1
    
    # Exploration (for algorithms that use epsilon-greedy)
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.5
    
    # Evaluation
    eval_every: int = 10_000  # Evaluate every N timesteps
    eval_episodes: int = 5    # Number of episodes to run for evaluation
    eval_deterministic: bool = True  # Use deterministic actions during evaluation
    
    # Agent configuration
    agent: AgentConfig = None


# Define available algorithms
ALGORITHMS = {
    "dqn": AgentConfig(
        name="dqn",
        module_path="jaxrl.agents.discrete.dqn",
        class_name="DQN",
        hyperparameters={
            "gamma": 0.99,
            "tau": 0.005,
        }
    ),
    # Add more algorithms here as they become available
    # "ddqn": AgentConfig(
    #     name="ddqn",
    #     module_path="jaxrl.agents.discrete.ddqn",
    #     class_name="DDQN",
    #     hyperparameters={
    #         "gamma": 0.99,
    #         "tau": 0.005,
    #     }
    # ),
}


def get_config(algorithm: str = "dqn", **overrides) -> ExperimentConfig:
    """
    Get a configuration for a specific algorithm with optional overrides.
    
    Args:
        algorithm: Name of the algorithm (e.g., "dqn")
        **overrides: Any parameters to override in the configuration
    
    Returns:
        ExperimentConfig: Complete configuration for the experiment
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithm '{algorithm}' not found. Available: {list(ALGORITHMS.keys())}")
    
    # Start with base config
    config = ExperimentConfig()
    config.agent = ALGORITHMS[algorithm]
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.agent, key):
            setattr(config.agent, key, value)
        elif key in config.agent.hyperparameters:
            config.agent.hyperparameters[key] = value
        else:
            # If it's not in the base config or agent config, add it to hyperparameters
            config.agent.hyperparameters[key] = value
    
    return config


# Predefined configurations for common setups
PREDEFINED_CONFIGS = {
    "dqn_cartpole": lambda: get_config(
        algorithm="dqn",
        env_id="CartPole-v1",
        total_timesteps=100_000,
        learning_rate=5e-4,
        batch_size=256,
    ),
    
    "dqn_atari": lambda: get_config(
        algorithm="dqn",
        env_id="ALE/Breakout-v5",
        total_timesteps=10_000_000,
        learning_rate=2.5e-4,
        batch_size=32,
        buffer_size=1_000_000,
        learning_starts=80_000,
        train_frequency=4,
        exploration_fraction=0.1,
        end_e=0.01,
    ),
}


def get_predefined_config(config_name: str) -> ExperimentConfig:
    """Get a predefined configuration by name."""
    if config_name not in PREDEFINED_CONFIGS:
        raise ValueError(f"Config '{config_name}' not found. Available: {list(PREDEFINED_CONFIGS.keys())}")
    return PREDEFINED_CONFIGS[config_name]() 