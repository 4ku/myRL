import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

class ReplayBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.observations = np.empty((capacity,) + obs_shape, dtype=np.float32)
        self.actions = np.empty((capacity,), dtype=np.int32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_observations = np.empty((capacity,) + obs_shape, dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, key):
        idx = jax.random.randint(key, (batch_size,), 0, self.size)
        # return FrozenDict
        return FrozenDict({
            "observations": jnp.array(self.observations[idx]),
            "actions": jnp.array(self.actions[idx]),
            "rewards": jnp.array(self.rewards[idx]),
            "next_observations": jnp.array(self.next_observations[idx]),
            "dones": jnp.array(self.dones[idx]),
        })
