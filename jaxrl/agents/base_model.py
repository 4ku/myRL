import flax
from typing import Union, Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from typing import Self

Array = Union[np.ndarray, jnp.ndarray]
Batch = Dict[str, Array]


class BaseModel(flax.struct.PyTreeNode):
    def update(self, batch: Batch, rng: jax.Array) -> Tuple[Self, dict]:
        raise NotImplementedError

    def sample_actions(self, observations: Array, rng: jax.Array) -> Array:
        raise NotImplementedError
