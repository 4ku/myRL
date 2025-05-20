import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from typing import Union, Dict
from flax.training.train_state import TrainState
import numpy as np
import chex

PRNGKey = int
Array = Union[np.ndarray, jnp.ndarray]
Batch = Dict[str, Array]


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(100)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(100)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(self.action_dim)(x)
        return x


class DQN(flax.struct.PyTreeNode):
    state: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observation: Array,
        action_dim: int,
        optimizer: optax.GradientTransformation,
        # hyperparameters
        gamma: float,
        tau: float,
        **kwargs,
    ) -> "DQN":
        q_network = QNetwork(action_dim=action_dim)
        q_state = TrainState.create(
            apply_fn=q_network.apply,
            params=q_network.init(rng, observation),
            target_params=q_network.init(rng, observation),
            tx=optimizer,
        )

        return cls(
            state=q_state,
            config=dict(
                gamma=gamma,
                tau=tau,
                **kwargs,
            ),
        )

    @jax.jit
    def update(self, batch: Batch):
        # check shapes
        batch_size = batch["observations"].shape[0]
        chex.assert_shape(batch["actions"], (batch_size,))
        chex.assert_shape(batch["rewards"], (batch_size,))
        chex.assert_shape(batch["dones"], (batch_size,))

        q_next_target = self.state.apply_fn(
            self.state.target_params, batch["next_observations"]
        )
        q_next_target = jnp.max(q_next_target, axis=-1)
        chex.assert_shape(q_next_target, (batch_size,))
        next_q_value = (
            batch["rewards"]
            + (1 - batch["dones"]) * self.config["gamma"] * q_next_target
        )
        chex.assert_shape(next_q_value, (batch_size,))

        def mse_loss(params):
            q_pred = self.state.apply_fn(params, batch["observations"])
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), batch["actions"].squeeze()]
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            self.state.params
        )
        new_state = self.state.apply_gradients(grads=grads)

        new_target_params = optax.incremental_update(self.state.params, self.state.target_params, self.config["tau"])
        new_state = new_state.replace(target_params=new_target_params)
        info = {
            "loss": loss_value,
            "q_pred": q_pred,
        }
        return self.replace(state=new_state), info

    @jax.jit
    def sample_actions(self, observations: Array):
        q_values = self.state.apply_fn(self.state.params, observations)
        return jnp.argmax(q_values, axis=-1)
