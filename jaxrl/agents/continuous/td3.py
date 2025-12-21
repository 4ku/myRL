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
    action_scale: float
    action_bias: float

    @classmethod
    def create(
        cls,
        rng: jax.Array,
        network: nn.Module,
        observation_sample: Array,
        action_space: gym.Space,
        optimizer: optax.GradientTransformation,
    ) -> "Actor":
        action_dim = action_space.shape[0]
        high = action_space.high
        low = action_space.low
        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0

        actor_def = DeterministicActor(network=network, action_dim=action_dim)
        params = actor_def.init(rng, observation_sample, training=False, rng=rng)

        state = TrainState.create(
            apply_fn=actor_def.apply, params=params, tx=optimizer, target_params=params
        )
        return cls(
            state=state,
            action_scale=action_scale,
            action_bias=action_bias,
        )

    def update(
        self, critic: "Critic", batch: Batch, rng: jax.Array
    ) -> Tuple["Actor", float]:
        observations = batch["observation"][:, 0]
        

        def actor_loss_fn(actor_params):
            actor_actions = self.state.apply_fn(
                actor_params, observations, training=True, rng=rng
            )

            # Use critic to evaluate actions
            def compute_q(params):
                return critic.state.apply_fn(
                    params, observations, actor_actions, training=False, rng=rng
                )

            # vmap over ensemble
            all_q_values = jax.vmap(compute_q)(critic.state.params)
            all_q_values = all_q_values.squeeze(-1)

            # Take median across ensemble
            q_median = jnp.median(all_q_values, axis=0)
            return -jnp.mean(q_median)

        grad_fn = jax.value_and_grad(actor_loss_fn)
        (loss), grads = grad_fn(self.state.params)
        new_state = self.state.apply_gradients(grads=grads)

        return self.replace(state=new_state), loss

    def soft_update(self, tau: float) -> "Actor":
        new_target_params = optax.incremental_update(
            self.state.params, self.state.target_params, tau
        )
        return self.replace(state=self.state.replace(target_params=new_target_params))


class Critic(flax.struct.PyTreeNode):
    state: TrainState

    @classmethod
    def create(
        cls,
        rng: jax.Array,
        network: nn.Module,
        observation_sample: Array,
        action_space: gym.Space,
        optimizer: optax.GradientTransformation,
        ensemble_size: int,
    ) -> "Critic":
        action_dim = action_space.shape[0]
        critic_def = ContinuousCritic(network=network)

        def init_critic(rng):
            return critic_def.init(
                rng,
                observation_sample,
                jnp.zeros((action_dim,)),
                training=False,
                rng=rng,
            )

        rngs = jax.random.split(rng, ensemble_size)
        params = jax.vmap(init_critic)(rngs)

        state = TrainState.create(
            apply_fn=critic_def.apply, params=params, tx=optimizer, target_params=params
        )
        return cls(state=state)

    def update(
        self, actor: Actor, batch: Batch, rng: jax.Array, config: dict
    ) -> Tuple["Critic", dict]:
        observations = batch["observation"][:, 0]
        next_observations = batch["observation"][:, 1]
        actions = batch["action"][:, 0]
        # Normalize actions
        actions = (actions - actor.action_bias) / actor.action_scale
        rewards = batch["reward"][:, 0]
        dones = batch["done"][:, 0]
        batch_size = rewards.shape[0]

        rng, noise_rng, subset_rng = jax.random.split(rng, 3)

        # -----------------------
        # Compute Target Q
        # -----------------------
        next_actions = actor.state.apply_fn(
            actor.state.target_params, next_observations, training=False, rng=rng
        )

        # Add noise to target actions
        noise = (
            jax.random.normal(noise_rng, (batch_size, config["action_dim"]))
            * config["policy_noise"]
        )
        noise = jnp.clip(noise, -config["noise_clip"], config["noise_clip"])
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        # Randomly sample subset of critics
        critic_subsample_size = config["critic_subsample_size"]
        subset_indices = jax.random.choice(
            subset_rng,
            config["critic_ensemble_size"],
            shape=(critic_subsample_size,),
            replace=False,
        )

        def compute_target_q(params):
            return self.state.apply_fn(
                params, next_observations, next_actions, training=False, rng=rng
            )

        sampled_target_params = jax.tree.map(
            lambda x: x[subset_indices], self.state.target_params
        )
        sampled_target_q = jax.vmap(compute_target_q)(sampled_target_params)
        chex.assert_shape(sampled_target_q, (critic_subsample_size, batch_size, 1))

        target_Q = jnp.min(sampled_target_q, axis=0).squeeze(-1)
        target_Q = rewards + (1 - dones) * config["gamma"] * target_Q

        # -----------------------
        # Update Critic
        # -----------------------
        def critic_loss_fn(critic_params):
            def compute_current_q(params):
                return self.state.apply_fn(
                    params, observations, actions, training=True, rng=rng
                )

            current_Q = jax.vmap(compute_current_q)(critic_params)
            current_Q = current_Q.squeeze(-1)

            loss = jnp.mean(jnp.square(current_Q - target_Q))
            return loss, current_Q

        grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        (loss, current_Q), grads = grad_fn(self.state.params)
        new_state = self.state.apply_gradients(grads=grads)

        info = {
            "critic_loss": loss,
            "q_values": current_Q,
            "target_q_values": target_Q,
            "critic_grad_norm": optax.global_norm(grads),
        }
        return self.replace(state=new_state), info

    def soft_update(self, tau: float) -> "Critic":
        new_target_params = optax.incremental_update(
            self.state.params, self.state.target_params, tau
        )
        return self.replace(state=self.state.replace(target_params=new_target_params))


class TD3TrainState(flax.struct.PyTreeNode):
    actor: Actor
    critic: Critic
    step: int = 0


class TD3(BaseModel):
    state: TD3TrainState
    config: dict = flax.struct.field(pytree_node=False)

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
        policy_freq: int,
        exploration_noise: float,
        critic_ensemble_size: int,
        critic_subsample_size: int,
    ) -> Self:
        action_dim = action_space.shape[0]

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        actor = Actor.create(
            rng=actor_rng,
            network=actor_network,
            observation_sample=observation_sample,
            action_space=action_space,
            optimizer=actor_optimizer,
        )

        critic = Critic.create(
            rng=critic_rng,
            network=critic_network,
            observation_sample=observation_sample,
            action_space=action_space,
            optimizer=critic_optimizer,
            ensemble_size=critic_ensemble_size,
        )

        return cls(
            state=TD3TrainState(
                actor=actor,
                critic=critic,
            ),
            config=dict(
                gamma=gamma,
                tau=tau,
                policy_noise=policy_noise,
                noise_clip=noise_clip,
                policy_freq=policy_freq,
                exploration_noise=exploration_noise,
                action_dim=action_dim,
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                action_bias=actor.action_bias,
                action_scale=actor.action_scale,
            ),
        )

    @jax.jit
    def update(self: Self, batch: Batch, rng: jax.Array) -> Tuple[Self, dict]:
        new_critic, critic_info = self.state.critic.update(
            actor=self.state.actor, batch=batch, rng=rng, config=self.config
        )

        # Soft update critic targets
        new_critic = new_critic.soft_update(self.config["tau"])

        should_update_actor = (self.state.step + 1) % self.config["policy_freq"] == 0

        def update_actor_step():
            new_actor, actor_loss = self.state.actor.update(new_critic, batch, rng)
            new_actor = new_actor.soft_update(self.config["tau"])
            return new_actor, actor_loss

        new_actor, actor_loss = jax.lax.cond(
            should_update_actor, update_actor_step, lambda: (self.state.actor, 0.0)
        )

        info = {**critic_info, "actor_loss": actor_loss}

        new_state = self.state.replace(
            actor=new_actor, critic=new_critic, step=self.state.step + 1
        )

        return self.replace(state=new_state), info

    @jax.jit
    def sample_actions(self: Self, observations: Array, rng: jax.Array) -> Array:
        actions = self.state.actor.state.apply_fn(
            self.state.actor.state.params, observations, training=False, rng=rng
        )
        actions = actions * self.state.actor.action_scale + self.state.actor.action_bias
        return actions
