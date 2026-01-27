import flax
import jax
import jax.numpy as jnp
import optax
from typing import Union, Dict, Tuple, Self
from flax.training.train_state import TrainState
import numpy as np
import chex
import flax.linen as nn

from jaxrl.networks.actor_critic_nets import DiscreteQFunction
from jaxrl.agents.base_model import BaseModel

Array = Union[np.ndarray, jnp.ndarray]
Batch = Dict[str, Array]


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class DQN(BaseModel):
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
        tau: float,
        ensemble_size: int,  # N: number of Q-networks in ensemble
        subsample_size: int,  # M: number of networks to sample for min
    ) -> Self:
        assert (
            subsample_size <= ensemble_size
        ), "subsample_size must be <= ensemble_size"

        critic = DiscreteQFunction(network=network, output_dim=action_dim)

        # Initialize N separate sets of parameters
        def init_single(rng):
            return critic.init(rng, observation_sample, training=False, rng=rng)

        rng, *init_rngs = jax.random.split(rng, ensemble_size + 1)
        init_rngs = jnp.stack(init_rngs)

        # vmap over initialization to get N parameter sets
        ensemble_params = jax.vmap(init_single)(init_rngs)

        return cls(
            state=TrainState.create(
                apply_fn=critic.apply,
                params=ensemble_params,
                target_params=ensemble_params,
                tx=optimizer,
            ),
            config=dict(
                gamma=gamma,
                tau=tau,
                critic_ensemble_size=ensemble_size,
                critic_subsample_size=subsample_size,
            ),
        )

    @jax.jit
    def update(self: Self, batch: Batch, rng: jax.Array) -> Tuple[Self, dict]:
        batch_size = batch["observation"].shape[0]
        observations = batch["observation"][:, 0]
        next_observations = batch["observation"][:, 1]
        actions = batch["action"][:, 0]
        rewards = batch["reward"][:, 0]
        dones = batch["done"][:, 0]
        chex.assert_shape(actions, (batch_size,))
        chex.assert_shape(rewards, (batch_size,))
        chex.assert_shape(dones, (batch_size,))

        ensemble_size = self.config["critic_ensemble_size"]
        subsample_size = self.config["critic_subsample_size"]

        subset_rng, new_rng = jax.random.split(rng, 2)

        # Randomly sample M indices from N critics
        subset_indices = jax.random.choice(
            subset_rng,
            ensemble_size,
            shape=(subsample_size,),
            replace=False,
        )

        def compute_q_values(params):
            return self.state.apply_fn(
                params, next_observations, training=False, rng=new_rng
            )

        sampled_target_params = jax.tree.map(
            lambda x: x[subset_indices], self.state.target_params
        )
        sampled_online_params = jax.tree.map(
            lambda x: x[subset_indices], self.state.params
        )
        sampled_target_q = jax.vmap(compute_q_values)(
            sampled_target_params
        )  # (M, batch_size, action_dim)
        sampled_online_q = jax.vmap(compute_q_values)(
            sampled_online_params
        )  # (M, batch_size, action_dim)

        median_sampled_online_q = jnp.median(
            sampled_online_q, axis=0
        )  # (batch_size, action_dim)
        best_actions = jnp.argmax(median_sampled_online_q, axis=-1)  # (batch_size,)
        sampled_q_for_actions = sampled_target_q[
            :, jnp.arange(batch_size), best_actions
        ]  # (M, batch_size)
        min_target_q = jnp.min(sampled_q_for_actions, axis=0)  # (batch_size,)

        next_q_value = rewards + (1 - dones) * self.config["gamma"] * min_target_q
        chex.assert_shape(next_q_value, (batch_size,))

        def compute_loss(p):
            q_pred = self.state.apply_fn(p, observations, training=True, rng=new_rng)
            q_pred = q_pred[jnp.arange(batch_size), actions]
            # return optax.huber_loss(q_pred, next_q_value).mean(), q_pred
            return optax.l2_loss(q_pred, next_q_value).mean(), q_pred

        compute_grads = jax.value_and_grad(compute_loss, has_aux=True)
        (all_losses, all_q_preds), grads = jax.vmap(compute_grads)(self.state.params)

        chex.assert_shape(all_losses, (ensemble_size,))
        chex.assert_shape(all_q_preds, (ensemble_size, batch_size))

        new_state = self.state.apply_gradients(grads=grads)

        # Update target params with polyak averaging
        new_target_params = optax.incremental_update(
            new_state.params, self.state.target_params, self.config["tau"]
        )
        new_state = new_state.replace(target_params=new_target_params)

        info = {
            "loss": all_losses,
            "q_pred": all_q_preds,
            "grad_norm": optax.global_norm(grads),
        }
        return self.replace(state=new_state), info

    @jax.jit
    def sample_actions(
        self: Self, observations: Array, rng: jax.Array, argmax: bool
    ) -> Tuple[Array, Array]:
        # Use median Q-values across ensemble for action selection
        def compute_q(params):
            return self.state.apply_fn(params, observations, training=False, rng=rng)

        all_q = jax.vmap(compute_q)(self.state.params)  # (N, batch_size, action_dim)
        median_q = jnp.median(all_q, axis=0)  # (batch_size, action_dim)
        return jnp.argmax(median_q, axis=-1), None
