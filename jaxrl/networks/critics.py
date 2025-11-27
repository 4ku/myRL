import flax.linen as nn
import jax.numpy as jnp
import jax


class DiscreteCritic(nn.Module):
    network: nn.Module
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool, rng: jax.Array) -> jnp.ndarray:
        x = self.network(x, training=training, rng=rng)
        return nn.Dense(self.output_dim)(x)