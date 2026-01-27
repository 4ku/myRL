import flax
import jax
import jax.numpy as jnp
import optax
from typing import Union, Dict, Tuple, Self, Optional
from flax.training.train_state import TrainState as BaseTrainState
import numpy as np
import flax.linen as nn
import gymnasium as gym
import chex
from functools import partial

from jaxrl.agents.base_model import BaseModel
from jaxrl.networks.actor_critic_nets import ContinuousQFunction, GaussianActor

Array = Union[np.ndarray, jnp.ndarray]
Batch = Dict[str, Array]


class TrainState(BaseTrainState):
    target_params: Optional[flax.core.FrozenDict] = None


class Temperature(flax.struct.PyTreeNode):
    state: TrainState
    target_entropy: float

    @classmethod
    def create(
        cls,
        optimizer: optax.GradientTransformation,
        target_entropy: float,
        init_value: float,
    ) -> "Temperature":
        log_alpha = jnp.log(init_value)

        def apply_fn(params):
            return jnp.exp(params["log_alpha"])

        state = TrainState.create(
            apply_fn=apply_fn,
            params={"log_alpha": log_alpha},
            tx=optimizer,
        )
        return cls(state=state, target_entropy=target_entropy)

    @jax.jit
    def update(self, entropy: float) -> Tuple["Temperature", dict]:
        def temperature_loss_fn(temp_params):
            log_alpha = temp_params["log_alpha"]
            loss = log_alpha * (entropy - self.target_entropy)
            return loss

        grad_fn = jax.value_and_grad(temperature_loss_fn)
        loss, grads = grad_fn(self.state.params)
        new_state = self.state.apply_gradients(grads=grads)

        info = {
            "alpha_loss": loss,
            "alpha": jnp.exp(new_state.params["log_alpha"]),
            "target_entropy": self.target_entropy,
        }
        return self.replace(state=new_state), info


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
        log_std_min: float,
        log_std_max: float,
    ) -> "Actor":
        action_dim = action_space.shape[0]
        high = action_space.high
        low = action_space.low
        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0

        actor_def = GaussianActor(
            network=network,
            action_dim=action_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        params = actor_def.init(rng, observation_sample, training=True, rng=rng)

        state = TrainState.create(apply_fn=actor_def.apply, params=params, tx=optimizer)
        return cls(
            state=state,
            config=dict(
                action_scale=action_scale,
                action_bias=action_bias,
            ),
        )

    @partial(jax.jit, static_argnames=("argmax",))
    def predict(
        self,
        observations: Array,
        rng: jax.Array,
        params: flax.core.FrozenDict | None = None,
        argmax: bool = False,
    ) -> Tuple[Array, Array]:
        params = params or self.state.params
        mean, log_std = self.state.apply_fn(
            params, observations, training=False, rng=rng
        )

        std = 0.0 if argmax else jnp.exp(log_std)

        noise = jax.random.normal(rng, mean.shape)
        u = mean + std * noise

        # Apply tanh squashing
        action = jnp.tanh(u)

        # Log prob calculation
        # log_prob(u) - log_det_jacobian
        log_prob = -0.5 * jnp.square(noise) - log_std - 0.5 * jnp.log(2 * jnp.pi)
        log_prob = jnp.sum(log_prob, axis=-1)

        # Enforce Tanh correction
        # log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
        log_det_jacobian = 2 * (jnp.log(2) - u - jax.nn.softplus(-2 * u))
        log_det_jacobian = jnp.sum(log_det_jacobian, axis=-1)

        log_prob -= log_det_jacobian

        return action, log_prob

    @jax.jit
    def update(
        self, critic: "Critic", alpha: float, batch: Batch, rng: jax.Array
    ) -> Tuple["Actor", dict]:
        observations = batch["observation"][:, 0]

        def actor_loss_fn(actor_params):
            actions, log_probs = self.predict(observations, rng, params=actor_params)

            # Get Q values
            q_values = critic.predict(
                observations, actions, rng
            )  # Shape: (ensemble, batch)
            chex.assert_shape(
                q_values,
                (critic.config["ensemble_size"], batch["observation"].shape[0]),
            )
            min_q = jnp.min(q_values, axis=0)  # Shape: (batch,)
            chex.assert_shape(min_q, (batch["observation"].shape[0],))

            # Loss = alpha * log_pi - Q
            actor_loss = jnp.mean(alpha * log_probs - min_q)
            return actor_loss, (log_probs, min_q)

        grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (loss, (log_probs, min_q)), grads = grad_fn(self.state.params)
        new_state = self.state.apply_gradients(grads=grads)

        info = {
            "actor_loss": loss,
            "actor_log_probs": log_probs,
            "actor_q_values": min_q,
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
        ensemble_size: int,  # N: number of Q-networks in ensemble
        subsample_size: int,  # M: number of networks to sample for min
    ) -> "Critic":
        assert (
            subsample_size <= ensemble_size
        ), "subsample_size must be <= ensemble_size"

        critic_def = ContinuousQFunction(network=network)

        def init_critic(rng):
            return critic_def.init(
                rng,
                observation_sample,
                action_space.sample(),
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
            ),
        )

    @jax.jit
    def predict(
        self,
        observations: Array,
        actions: Array,
        rng: jax.Array,
        params: flax.core.FrozenDict | None = None,
    ) -> Array:  # shape: (ensemble, batch)
        params = params or self.state.params
        q_values = jax.vmap(
            lambda p: self.state.apply_fn(
                p, observations, actions, training=False, rng=rng
            )
        )(params)
        return q_values.squeeze(-1)

    @jax.jit
    def update(
        self, actor: Actor, alpha: float, batch: Batch, rng: jax.Array
    ) -> Tuple["Critic", dict]:
        observations = batch["observation"][:, 0]
        next_observations = batch["observation"][:, 1]
        actions = batch["action"][:, 0]
        # Normalize actions
        actions = (actions - actor.config["action_bias"]) / actor.config["action_scale"]
        rewards = batch["reward"][:, 0]
        dones = batch["done"][:, 0]

        # Sample next actions
        rng, next_action_rng = jax.random.split(rng)
        next_actions, next_log_probs = actor.predict(next_observations, next_action_rng)

        # Compute target Q
        # Use target params for critic

        target_q_values = self.predict(
            next_observations, next_actions, rng, params=self.state.target_params
        )
        chex.assert_shape(
            target_q_values,
            (self.config["ensemble_size"], batch["observation"].shape[0]),
        )
        min_target_q = jnp.min(target_q_values, axis=0)
        chex.assert_shape(min_target_q, (batch["observation"].shape[0],))

        target_q = rewards + self.config["gamma"] * (1 - dones) * (
            min_target_q - alpha * next_log_probs
        )

        def critic_loss_fn(critic_params):
            current_q_values = jax.vmap(
                lambda p: self.state.apply_fn(
                    p, observations, actions, training=True, rng=rng
                )
            )(critic_params).squeeze(-1)

            # Loss for each critic
            loss = jnp.mean(jnp.square(current_q_values - target_q))
            return loss, current_q_values

        grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        (loss, current_q_values), grads = grad_fn(self.state.params)
        new_state = self.state.apply_gradients(grads=grads)

        # Polyak averaging for target params
        new_target_params = optax.incremental_update(
            new_state.params, self.state.target_params, self.config["tau"]
        )
        new_state = new_state.replace(target_params=new_target_params)

        info = {
            "critic_loss": loss,
            "q_values": current_q_values,
            "target_q_values": target_q_values,
            "critic_grad_norm": optax.global_norm(grads),
        }
        return self.replace(state=new_state), info


class SACTrainState(flax.struct.PyTreeNode):
    actor: Actor
    critic: Critic
    temp: Temperature


class SAC(BaseModel):
    state: SACTrainState

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
        temperature_optimizer: optax.GradientTransformation,
        critic_ensemble_size: int,
        critic_subsample_size: int,
        actor_log_std_min: float,
        actor_log_std_max: float,
        init_temperature: float,
        # hyperparameters
        gamma: float,
        tau: float,
        target_entropy: float,
    ) -> Self:
        assert (
            critic_subsample_size <= critic_ensemble_size
        ), "critic_subsample_size must be <= critic_ensemble_size"

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        actor = Actor.create(
            rng=actor_rng,
            network=actor_network,
            observation_sample=observation_sample,
            action_space=action_space,
            optimizer=actor_optimizer,
            log_std_min=actor_log_std_min,
            log_std_max=actor_log_std_max,
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
        )

        temp = Temperature.create(
            optimizer=temperature_optimizer,
            target_entropy=target_entropy,
            init_value=init_temperature,
        )

        return cls(
            state=SACTrainState(
                actor=actor,
                critic=critic,
                temp=temp,
            ),
        )

    @jax.jit
    def update(self: Self, batch: Batch, rng: jax.Array) -> Tuple[Self, dict]:
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        # Get current alpha
        alpha = self.state.temp.state.apply_fn(self.state.temp.state.params)

        # Update Critic
        new_critic, critic_info = self.state.critic.update(
            actor=self.state.actor, alpha=alpha, batch=batch, rng=critic_rng
        )

        # Update Actor
        new_actor, actor_info = self.state.actor.update(
            critic=new_critic,
            alpha=alpha,
            batch=batch,
            rng=actor_rng,
        )

        # Update Temperature
        entropy = -jnp.mean(actor_info["actor_log_probs"])
        new_temp, temp_info = self.state.temp.update(entropy)

        new_state = self.state.replace(
            actor=new_actor, critic=new_critic, temp=new_temp
        )

        info = {**critic_info, **actor_info, **temp_info}

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self: Self, observations: Array, rng: jax.Array, argmax: bool
    ) -> Tuple[Array, Array]:
        actions, log_probs = self.state.actor.predict(observations, rng, argmax=argmax)
        # Scale normalized actions [-1, 1] to the environment action space.
        actions = (
            actions * self.state.actor.config["action_scale"]
            + self.state.actor.config["action_bias"]
        )
        return actions, log_probs
