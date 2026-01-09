import flax
import jax
import jax.numpy as jnp
import optax
from typing import Union, Dict, Tuple, Self
from flax.training.train_state import TrainState
import numpy as np
import flax.linen as nn
import chex
import gymnasium as gym

from jaxrl.agents.base_model import BaseModel
from jaxrl.networks.actor_critic_nets import GaussianActor

Array = Union[np.ndarray, jnp.ndarray]
Batch = Dict[str, Array]


class VPG(BaseModel):
    state: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        rng: jax.Array,
        observation_sample: Array,
        action_space: gym.Space,
        network: nn.Module,
        optimizer: optax.GradientTransformation,
        # hyperparameters
        gamma: float,
        ent_coef: float,
        log_std_min: float,
        log_std_max: float,
    ) -> Self:
        action_dim = action_space.shape[0]
        actor = GaussianActor(
            network=network,
            action_dim=action_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

        rng, init_rng = jax.random.split(rng)
        params = actor.init(init_rng, observation_sample, training=True, rng=init_rng)

        # Action bounding
        high = action_space.high
        low = action_space.low
        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0

        return cls(
            state=TrainState.create(
                apply_fn=actor.apply,
                params=params,
                tx=optimizer,
            ),
            config=dict(
                gamma=gamma,
                action_dim=action_dim,
                action_scale=action_scale,
                action_bias=action_bias,
                ent_coef=ent_coef,
            ),
        )

    @jax.jit
    def update(self: Self, batch: Batch, rng: jax.Array) -> Tuple[Self, dict]:
        batch_size = batch["observation"].shape[0]
        observations = batch["observation"][:, 0]
        actions = batch["action"][:, 0]
        returns = batch["return"][:, 0]

        # Normalize returns
        returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)

        chex.assert_shape(actions, (batch_size, self.config["action_dim"]))
        chex.assert_shape(returns, (batch_size,))

        def compute_loss(params):
            mean, log_std = self.state.apply_fn(
                params, observations, training=True, rng=rng
            )
            std = jnp.exp(log_std)
            chex.assert_shape(mean, (batch_size, self.config["action_dim"]))
            chex.assert_shape(std, (batch_size, self.config["action_dim"]))

            # Unsquash actions to get raw actions (before tanh)
            action_scale = self.config["action_scale"]
            action_bias = self.config["action_bias"]

            normalized_action = (actions - action_bias) / action_scale
            normalized_action = jnp.clip(normalized_action, -1 + 1e-6, 1 - 1e-6)
            raw_actions = jnp.arctanh(normalized_action)

            # Gaussian log prob of raw actions
            log_probs_all = (
                -0.5 * jnp.square((raw_actions - mean) / std)
                - log_std
                - 0.5 * jnp.log(2 * jnp.pi)
            )
            log_prob_raw = jnp.sum(log_probs_all, axis=-1)

            # Correction for tanh squashing
            # log p(a) = log p(u) - sum(log(scale * (1 - tanh(u)^2)))
            # 1 - tanh(u)^2 = 1 - normalized_action^2
            log_det_jacobian = jnp.sum(
                jnp.log(action_scale) + jnp.log(1 - jnp.square(normalized_action)),
                axis=-1,
            )

            log_probs = log_prob_raw - log_det_jacobian

            entropy = -jnp.mean(log_probs)

            loss = -jnp.mean(log_probs * returns) - self.config["ent_coef"] * entropy

            return loss, (log_probs, entropy)

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, (log_probs, entropy)), grads = grad_fn(self.state.params)

        new_state = self.state.apply_gradients(grads=grads)

        info = {
            "loss": loss,
            "loss_returns": -jnp.mean(log_probs * returns),
            "loss_entropy": -self.config["ent_coef"] * entropy,
            "log_probs": log_probs,
            "entropy": entropy,
            "returns": returns,
            "grad_norm": optax.global_norm(grads),
        }

        return self.replace(state=new_state), info

    @jax.jit
    def sample_actions(self: Self, observations: Array, rng: jax.Array) -> Array:
        mean, log_std = self.state.apply_fn(
            self.state.params, observations, training=False, rng=rng
        )
        std = jnp.exp(log_std)
        raw_actions = mean + std * jax.random.normal(rng, shape=mean.shape)

        actions = (
            jnp.tanh(raw_actions) * self.config["action_scale"]
            + self.config["action_bias"]
        )

        return actions
