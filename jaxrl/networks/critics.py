import flax.linen as nn
import jax.numpy as jnp


class DiscreteCritic(nn.Module):
    network: nn.Module
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        x = self.network(x, training=training)
        return nn.Dense(self.output_dim)(x)