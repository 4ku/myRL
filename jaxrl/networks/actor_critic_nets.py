import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Tuple


class GaussianActor(nn.Module):
    network: nn.Module
    action_dim: int
    log_std_min: float
    log_std_max: float

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


class DeterministicActor(nn.Module):
    network: nn.Module
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool, rng: jax.Array) -> jnp.ndarray:
        x = self.network(x, training=training, rng=rng)
        x = nn.Dense(self.action_dim)(x)
        return nn.tanh(x)


class DiscreteQFunction(nn.Module):
    network: nn.Module
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool, rng: jax.Array) -> jnp.ndarray:
        x = self.network(x, training=training, rng=rng)
        return nn.Dense(self.output_dim)(x)


class ContinuousQFunction(nn.Module):
    network: nn.Module

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray, action: jnp.ndarray, training: bool, rng: jax.Array
    ) -> jnp.ndarray:
        x = jnp.concatenate([obs, action], axis=-1)
        x = self.network(x, training=training, rng=rng)
        return nn.Dense(1)(x)


class ValueFunction(nn.Module):
    network: nn.Module

    @nn.compact
    def __call__(self, obs: jnp.ndarray, training: bool, rng: jax.Array) -> jnp.ndarray:
        x = self.network(obs, training=training, rng=rng)
        return nn.Dense(1)(x)
