import flax
import jax
import jax.numpy as jnp
import optax
from typing import Union, Dict, Tuple, Self
from flax.training.train_state import TrainState
import numpy as np
import chex
import flax.linen as nn

from jaxrl.networks.critics import DiscreteCritic
from jaxrl.agents.base_model import BaseModel

Array = Union[np.ndarray, jnp.ndarray]
Batch = Dict[str, Array]


class TrainState(TrainState):
    target_params: flax.core.FrozenDict
    rng: jax.Array


class DQN(BaseModel):
    state: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        rng: jax.Array,
        observation_sample: Array,
        action_sample: Array,
        network: nn.Module,
        optimizer: optax.GradientTransformation,
        # hyperparameters
        gamma: float,
        tau: float,
    ) -> Self:
        if len(action_sample.shape) > 1:
            raise ValueError("Action sample must be a 1D array")
        output_dim = action_sample.shape[0] if len(action_sample.shape) == 1 else 1
        critic = DiscreteCritic(network=network, output_dim=output_dim)

        rng, init_rng, target_rng = jax.random.split(rng, 3)
        return cls(
            state=TrainState.create(
                apply_fn=critic.apply,
                params=critic.init(init_rng, observation_sample, training=False),
                target_params=critic.init(target_rng, observation_sample, training=False),
                tx=optimizer,
                rng=rng,
            ),
            config=dict(gamma=gamma, tau=tau),
        )

    @jax.jit
    def update(self: Self, batch: Batch) -> Tuple[Self, dict]:
        # check shapes
        batch_size = batch["observation"].shape[0]
        observations = batch["observation"][:, 0]
        next_observations = batch["observation"][:, 1]
        actions = batch["action"][:, 0]
        rewards = batch["reward"][:, 0]
        dones = batch["done"][:, 0]
        chex.assert_shape(actions, (batch_size,))
        chex.assert_shape(rewards, (batch_size,))
        chex.assert_shape(dones, (batch_size,))

        q_next_target = self.state.apply_fn(
            self.state.target_params, next_observations, training=False
        )  # (batch_size, action_dim)
        q_next_target = jnp.max(q_next_target, axis=-1)
        chex.assert_shape(q_next_target, (batch_size,))

        next_q_value = (
            rewards
            + (1 - dones) * self.config["gamma"] * q_next_target
        )
        chex.assert_shape(next_q_value, (batch_size,))

        new_rng, _ = jax.random.split(self.state.rng, 2)

        def mse_loss(params):
            q_pred = self.state.apply_fn(
                params, observations, training=True, rngs=new_rng
            )  # (batch_size, action_dim)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions]
            chex.assert_shape(q_pred, (batch_size,))
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            self.state.params
        )
        new_state = self.state.apply_gradients(grads=grads)
        new_state = new_state.replace(rng=new_rng)

        new_target_params = optax.incremental_update(
            self.state.params, self.state.target_params, self.config["tau"]
        )
        new_state = new_state.replace(target_params=new_target_params)
        info = {
            "loss": loss_value,
            "q_pred": q_pred,
        }
        return self.replace(state=new_state), info

    @jax.jit
    def sample_actions(self: Self, observations: Array) -> Array:
        q_values = self.state.apply_fn(self.state.params, observations, training=False)
        return jnp.argmax(q_values, axis=-1)
