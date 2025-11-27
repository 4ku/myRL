import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Callable
import jax


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray]
    use_layer_norm: bool
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool, rng: jax.Array) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)

            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

            x = self.activation(x)

            if self.dropout_rate > 0:
                layer_rng = jax.random.fold_in(rng, i)
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x, rng=layer_rng)

        return x
