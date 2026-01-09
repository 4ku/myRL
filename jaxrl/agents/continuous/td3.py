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
from jaxrl.networks.actor_critic_nets import ContinuousCritic, DeterministicActor

Array = Union[np.ndarray, jnp.ndarray]
Batch = Dict[str, Array]


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class Actor(flax.struct.PyTreeNode):
    state: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        rng: jax.Array,
        network: nn.Module,
        observation_sample: Array,
        action_space: gym.Space,
        optimizer: optax.GradientTransformation,
        # hyperparameters
        tau: float,
    ) -> "Actor":
        action_dim = action_space.shape[0]
        high = action_space.high
        low = action_space.low
        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0

        actor_def = DeterministicActor(network=network, action_dim=action_dim)
        params = actor_def.init(rng, observation_sample, training=True, rng=rng)

        state = TrainState.create(
            apply_fn=actor_def.apply, params=params, tx=optimizer, target_params=params
        )
        return cls(
            state=state,
            config=dict(
                tau=tau,
                action_scale=action_scale,
                action_bias=action_bias,
            ),
        )

    @jax.jit
    def predict(
        self,
        observations: Array,
        rng: jax.Array,
        params: flax.core.FrozenDict | None = None,
    ) -> Array:
        params = params or self.state.params
        return self.state.apply_fn(params, observations, training=False, rng=rng)

    @jax.jit
    def update(
        self, critic: "Critic", batch: Batch, rng: jax.Array
    ) -> Tuple["Actor", float]:
        observations = batch["observation"][:, 0]

        def actor_loss_fn(actor_params):
            actor_actions = self.state.apply_fn(
                actor_params, observations, training=True, rng=rng
            )
            q_median = critic.predict(observations, actor_actions, rng)
            return -jnp.mean(q_median), q_median

        grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (loss, q_median), grads = grad_fn(self.state.params)
        new_state = self.state.apply_gradients(grads=grads)

        new_target_params = optax.incremental_update(
            new_state.params, self.state.target_params, self.config["tau"]
        )
        new_state = new_state.replace(target_params=new_target_params)

        info = {
            "actor_loss": loss,
            "actor_predicted_q": q_median,
            "actor_grad_norm": optax.global_norm(grads),
        }
        return self.replace(state=new_state), info


class Critic(flax.struct.PyTreeNode):
    state: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        rng: jax.Array,
        network: nn.Module,
        observation_sample: Array,
        action_space: gym.Space,
        optimizer: optax.GradientTransformation,
        # hyperparameters
        gamma: float,
        tau: float,
        ensemble_size: int,
        subsample_size: int,
        policy_noise: float,
        noise_clip: float,
    ) -> "Critic":
        assert (
            subsample_size <= ensemble_size
        ), "subsample_size must be <= ensemble_size"

        action_dim = action_space.shape[0]

        critic_def = ContinuousCritic(network=network)

        def init_critic(rng):
            return critic_def.init(
                rng,
                observation_sample,
                jnp.zeros((action_dim,)),
                training=True,
                rng=rng,
            )

        rngs = jax.random.split(rng, ensemble_size)
        params = jax.vmap(init_critic)(rngs)

        state = TrainState.create(
            apply_fn=critic_def.apply, params=params, tx=optimizer, target_params=params
        )
        return cls(
            state=state,
            config=dict(
                gamma=gamma,
                tau=tau,
                ensemble_size=ensemble_size,
                subsample_size=subsample_size,
                policy_noise=policy_noise,
                noise_clip=noise_clip,
                action_dim=action_dim,
            ),
        )

    @jax.jit
    def predict(
        self,
        observations: Array,
        actions: Array,
        rng: jax.Array,
        params: flax.core.FrozenDict | None = None,
    ) -> Array:
        if params is None:
            q_values = jax.vmap(
                lambda p: self.state.apply_fn(
                    p, observations, actions, training=False, rng=rng
                )
            )(self.state.params)
            return jnp.median(q_values, axis=0)

        return self.state.apply_fn(
            params, observations, actions, training=False, rng=rng
        )

    @jax.jit
    def update(
        self, actor: Actor, batch: Batch, rng: jax.Array
    ) -> Tuple["Critic", dict]:
        observations = batch["observation"][:, 0]
        next_observations = batch["observation"][:, 1]
        actions = batch["action"][:, 0]
        # Normalize actions
        actions = (actions - actor.config["action_bias"]) / actor.config["action_scale"]
        rewards = batch["reward"][:, 0]
        dones = batch["done"][:, 0]
        batch_size = rewards.shape[0]

        rng, noise_rng, subset_rng, actor_rng, loss_rng = jax.random.split(rng, 5)

        # -----------------------
        # Compute Target Q
        # -----------------------
        next_actions = actor.predict(next_observations, actor_rng)

        # Add noise to target actions
        noise = (
            jax.random.normal(noise_rng, (batch_size, self.config["action_dim"]))
            * self.config["policy_noise"]
        )
        noise = jnp.clip(noise, -self.config["noise_clip"], self.config["noise_clip"])
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        # Randomly sample subset of critics
        subsample_size = self.config["subsample_size"]
        subset_indices = jax.random.choice(
            subset_rng,
            self.config["ensemble_size"],
            shape=(subsample_size,),
            replace=False,
        )

        sampled_target_params = jax.tree.map(
            lambda x: x[subset_indices], self.state.target_params
        )
        sampled_target_q = jax.vmap(self.predict, in_axes=(None, None, None, 0))(
            next_observations, next_actions, rng, sampled_target_params
        )
        chex.assert_shape(sampled_target_q, (subsample_size, batch_size, 1))

        target_Q = jnp.min(sampled_target_q, axis=0).squeeze(-1)
        target_Q = rewards + (1 - dones) * self.config["gamma"] * target_Q
        chex.assert_shape(target_Q, (batch_size,))

        # -----------------------
        # Update Critic
        # -----------------------
        def critic_loss_fn(critic_params):
            def compute_current_q(params):
                return self.state.apply_fn(
                    params, observations, actions, training=True, rng=loss_rng
                )

            current_Q = jax.vmap(compute_current_q)(critic_params)
            current_Q = current_Q.squeeze(-1)

            loss = jnp.mean(jnp.square(current_Q - target_Q))
            return loss, current_Q

        grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        (loss, current_Q), grads = grad_fn(self.state.params)
        new_state = self.state.apply_gradients(grads=grads)

        # Update target params with polyak averaging
        new_target_params = optax.incremental_update(
            new_state.params, self.state.target_params, self.config["tau"]
        )
        new_state = new_state.replace(target_params=new_target_params)

        info = {
            "critic_loss": loss,
            "q_values": current_Q,
            "target_q_values": target_Q,
            "critic_grad_norm": optax.global_norm(grads),
        }
        return self.replace(state=new_state), info


class TD3TrainState(flax.struct.PyTreeNode):
    actor: Actor
    critic: Critic


class TD3(BaseModel):
    state: TD3TrainState
    policy_freq: int
    step: int = 0

    @classmethod
    def create(
        cls,
        rng: jax.Array,
        observation_sample: Array,
        action_space: gym.Space,
        actor_network: nn.Module,
        critic_network: nn.Module,
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        # hyperparameters
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        critic_ensemble_size: int,
        critic_subsample_size: int,
        policy_freq: int,
    ) -> Self:
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        actor = Actor.create(
            rng=actor_rng,
            network=actor_network,
            observation_sample=observation_sample,
            action_space=action_space,
            optimizer=actor_optimizer,
            tau=tau,
        )

        critic = Critic.create(
            rng=critic_rng,
            network=critic_network,
            observation_sample=observation_sample,
            action_space=action_space,
            optimizer=critic_optimizer,
            gamma=gamma,
            tau=tau,
            ensemble_size=critic_ensemble_size,
            subsample_size=critic_subsample_size,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
        )

        return cls(
            state=TD3TrainState(
                actor=actor,
                critic=critic,
            ),
            policy_freq=policy_freq,
            step=0,
        )

    def update(self: Self, batch: Batch, rng: jax.Array) -> Tuple[Self, dict]:
        new_critic, critic_info = self.state.critic.update(
            actor=self.state.actor, batch=batch, rng=rng
        )
        if self.step % self.policy_freq == 0:
            new_actor, actor_info = self.state.actor.update(new_critic, batch, rng)
            info = {**critic_info, **actor_info}
            new_state = self.state.replace(actor=new_actor, critic=new_critic)
            return self.replace(state=new_state, step=self.step + 1), info
        else:
            new_state = self.state.replace(critic=new_critic)
            return self.replace(state=new_state, step=self.step + 1), critic_info

    @jax.jit
    def sample_actions(self: Self, observations: Array, rng: jax.Array) -> Array:
        actions = self.state.actor.predict(observations, rng)
        # Scale actions to environment space
        return (
            actions * self.state.actor.config["action_scale"]
            + self.state.actor.config["action_bias"]
        )
