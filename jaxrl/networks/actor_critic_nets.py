import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Tuple

class GaussianActor(nn.Module):
    network: nn.Module
    action_dim: int
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool, rng: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = self.network(x, training=training, rng=rng)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

class DiscreteActor(nn.Module):
    network: nn.Module
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool, rng: jax.Array) -> jnp.ndarray:
        x = self.network(x, training=training, rng=rng)
        logits = nn.Dense(self.action_dim)(x)
        return logits


class DiscreteCritic(nn.Module):
    network: nn.Module
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool, rng: jax.Array) -> jnp.ndarray:
        x = self.network(x, training=training, rng=rng)
        return nn.Dense(self.output_dim)(x)
