import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Callable


class MLP(nn.Module):
    hidden_dims: Sequence[int] = (128, 128)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    use_layer_norm: bool = True
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        for size in self.hidden_dims:
            x = nn.Dense(size)(x)

            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

            x = self.activation(x)

            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x
