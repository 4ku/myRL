import flax
import jax
import jax.numpy as jnp
import optax
from typing import Union, Dict, Tuple, Self
from flax.training.train_state import TrainState
import numpy as np
import flax.linen as nn
import chex

from jaxrl.agents.base_model import BaseModel
from jaxrl.networks.actor_critic_nets import DiscreteActor, GaussianActor

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
        action_dim: int,
        network: nn.Module,
        optimizer: optax.GradientTransformation,
        # hyperparameters
        gamma: float,
        is_continuous: bool,
    ) -> Self:
        if is_continuous:
            actor = GaussianActor(network=network, action_dim=action_dim)
        else:
            actor = DiscreteActor(network=network, action_dim=action_dim)

        rng, init_rng = jax.random.split(rng)
        params = actor.init(init_rng, observation_sample, training=False, rng=init_rng)

        return cls(
            state=TrainState.create(
                apply_fn=actor.apply,
                params=params,
                tx=optimizer,
            ),
            config=dict(gamma=gamma, is_continuous=is_continuous),
        )

    @jax.jit
    def update(self: Self, batch: Batch, rng: jax.Array) -> Tuple[Self, dict]:
        batch_size = batch["observation"].shape[0]
        observations = batch["observation"][:, 0]
        actions = batch["action"][:, 0]
        returns = batch["return"][:, 0]
        chex.assert_shape(actions, (batch_size,))
        chex.assert_shape(returns, (batch_size,))
        
        if not self.config["is_continuous"]:
            actions = actions.astype(jnp.int32)

        def compute_loss(params):
            if self.config["is_continuous"]:
                mean, log_std = self.state.apply_fn(
                    params, observations, training=True, rng=rng
                )
                std = jnp.exp(log_std)

                # Gaussian log prob: -0.5 * ((x - mu)/sigma)^2 - log(sigma) - 0.5 * log(2pi)
                # Sum over action dimension
                log_probs_all = (
                    -0.5 * jnp.square((actions - mean) / std)
                    - log_std
                    - 0.5 * jnp.log(2 * jnp.pi)
                )
                log_probs = jnp.sum(log_probs_all, axis=-1)

            else:
                logits = self.state.apply_fn(
                    params, observations, training=True, rng=rng
                )
                log_probs_all = jax.nn.log_softmax(logits)
                # Select log probability of the taken action
                log_probs = jnp.take_along_axis(
                    log_probs_all, actions[:, None], axis=1
                ).squeeze()

            loss = -jnp.mean(log_probs * returns)
            return loss, log_probs

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, log_probs), grads = grad_fn(self.state.params)

        new_state = self.state.apply_gradients(grads=grads)

        info = {
            "loss": loss,
            "log_probs": log_probs,
            "returns": returns,
            "grad_norm": optax.global_norm(grads),
        }

        return self.replace(state=new_state), info

    @jax.jit
    def sample_actions(self: Self, observations: Array, rng: jax.Array) -> Array:
        if self.config["is_continuous"]:
            mean, log_std = self.state.apply_fn(
                self.state.params, observations, training=False, rng=rng
            )
            std = jnp.exp(log_std)
            actions = mean + std * jax.random.normal(rng, shape=mean.shape)
        else:
            logits = self.state.apply_fn(
                self.state.params, observations, training=False, rng=rng
            )
            actions = jax.random.categorical(rng, logits)

        return actions
