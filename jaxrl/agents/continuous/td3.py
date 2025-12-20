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

class TD3TrainState(flax.struct.PyTreeNode):
    actor: TrainState
    critic: TrainState
    target_actor_params: flax.core.FrozenDict
    target_critic_params: flax.core.FrozenDict
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
        
        # Action bounding
        high = action_space.high
        low = action_space.low
        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0

        actor = DeterministicActor(network=actor_network, action_dim=action_dim)
        critic = ContinuousCritic(network=critic_network)

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        
        # Initialize actor
        actor_params = actor.init(actor_rng, observation_sample, training=False, rng=actor_rng)
        
        # Initialize critic (ensemble of 2)
        def init_critic(rng):
            return critic.init(rng, observation_sample, jnp.zeros((action_dim,)), training=False, rng=rng)
            
        critic_rngs = jax.random.split(critic_rng, critic_ensemble_size)
        critic_params = jax.vmap(init_critic)(critic_rngs)

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=actor_optimizer,
        )
        
        critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=critic_optimizer,
        )

        return cls(
            state=TD3TrainState(
                actor=actor_state,
                critic=critic_state,
                target_actor_params=actor_params,
                target_critic_params=critic_params,
            ),
            config=dict(
                gamma=gamma,
                tau=tau,
                policy_noise=policy_noise,
                noise_clip=noise_clip,
                policy_freq=policy_freq,
                exploration_noise=exploration_noise,
                action_dim=action_dim,
                action_scale=action_scale,
                action_bias=action_bias,
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
            ),
        )

    @jax.jit
    def update(self: Self, batch: Batch, rng: jax.Array) -> Tuple[Self, dict]:
        batch_size = batch["observation"].shape[0]
        observations = batch["observation"][:, 0]
        next_observations = batch["observation"][:, 1]
        actions = batch["action"][:, 0]
        # Normalize actions to [-1, 1] as critic expects normalized inputs
        actions = (actions - self.config["action_bias"]) / self.config["action_scale"]
        rewards = batch["reward"][:, 0]
        dones = batch["done"][:, 0]
        
        rng, noise_rng = jax.random.split(rng)

        critic_ensemble_size = self.config["critic_ensemble_size"]
        critic_subsample_size = self.config["critic_subsample_size"]

        subset_rng, rng = jax.random.split(rng, 2)

        # Randomly sample M indices from N critics
        subset_indices = jax.random.choice(
            subset_rng,
            critic_ensemble_size,
            shape=(critic_subsample_size,),
            replace=False,
        )

        # -----------------------
        # Update Critic
        # -----------------------
        
        next_actions = self.state.actor.apply_fn(self.state.target_actor_params, next_observations, training=False, rng=rng)
        
        # Add noise to target actions
        noise = jax.random.normal(noise_rng, (batch_size, self.config["action_dim"])) * self.config["policy_noise"]
        noise = jnp.clip(noise, -self.config["noise_clip"], self.config["noise_clip"])
        next_actions = next_actions + noise
        next_actions = jnp.clip(next_actions, -1.0, 1.0)
        
        # Compute the target Q value
        def compute_target_q(params):
            return self.state.critic.apply_fn(params, next_observations, next_actions, training=False, rng=rng)
            
        sampled_target_params = jax.tree.map(
            lambda x: x[subset_indices], self.state.target_critic_params
        )
        sampled_target_q = jax.vmap(compute_target_q)(
            sampled_target_params
        )  # (M, batch_size, 1)
        chex.assert_shape(sampled_target_q, (critic_subsample_size, batch_size, 1))
        target_Q = jnp.min(sampled_target_q, axis=0).squeeze(-1) # (batch,)
        chex.assert_shape(target_Q, (batch_size,))
        target_Q = rewards + (1 - dones) * self.config["gamma"] * target_Q
        
        # Get current Q estimates
        def critic_loss_fn(critic_params):
            def compute_current_q(params):
                return self.state.critic.apply_fn(params, observations, actions, training=True, rng=rng)
            
            current_Q = jax.vmap(compute_current_q)(critic_params) # (critic_ensemble_size, batch, 1)
            chex.assert_shape(current_Q, (critic_ensemble_size, batch_size, 1))
            current_Q = current_Q.squeeze(-1) # (critic_ensemble_size, batch)
            
            # MSE loss for each critic
            loss = jnp.mean(jnp.square(current_Q - target_Q))
            return loss, current_Q

        grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        (critic_loss, current_Q), critic_grads = grad_fn(self.state.critic.params)
        
        new_critic_state = self.state.critic.apply_gradients(grads=critic_grads)
        
        # -----------------------
        # Update Actor (Delayed)
        # -----------------------
        
        # We need to use 'step' from state to check policy_freq
        # flax.struct.PyTreeNode is immutable, so we create new state at the end
        
        # Conditionally update actor
        should_update_actor = (self.state.step + 1) % self.config["policy_freq"] == 0
        
        def update_actor(actor_state, critic_params):
            def actor_loss_fn(actor_params):
                actor_actions = self.state.actor.apply_fn(actor_params, observations, training=True, rng=rng)
                
                # Compute Q-values for all critics to minimize variance
                def compute_q(params):
                    return self.state.critic.apply_fn(params, observations, actor_actions, training=False, rng=rng)
                
                # vmap over ensemble
                all_q_values = jax.vmap(compute_q)(critic_params) # (N, batch_size, 1)
                all_q_values = all_q_values.squeeze(-1) # (N, batch_size)
                chex.assert_shape(all_q_values, (critic_ensemble_size, batch_size))
                
                # Take median across ensemble
                q_median = jnp.median(all_q_values, axis=0) # (batch_size,)
                
                return -jnp.mean(q_median)

            actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
            new_actor_state = actor_state.apply_gradients(grads=actor_grads)
            return new_actor_state, actor_loss
        
        new_actor_state, actor_loss = jax.lax.cond(
            should_update_actor,
            lambda: update_actor(self.state.actor, new_critic_state.params),
            lambda: (self.state.actor, 0.0)
        )
        
        new_target_critic_params = optax.incremental_update(new_critic_state.params, self.state.target_critic_params, self.config["tau"])

        new_target_actor_params = jax.lax.cond(
            should_update_actor,
            lambda: optax.incremental_update(new_actor_state.params, self.state.target_actor_params, self.config["tau"]),
            lambda: self.state.target_actor_params
        )

        # Log info
        info = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "q_values": current_Q,
            "target_q_values": target_Q,
            "critic_grad_norm": optax.global_norm(critic_grads),
        }
        
        new_state = self.state.replace(
            actor=new_actor_state,
            critic=new_critic_state,
            target_actor_params=new_target_actor_params,
            target_critic_params=new_target_critic_params,
            step=self.state.step + 1
        )
        
        return self.replace(state=new_state), info

    @jax.jit
    def sample_actions(self: Self, observations: Array, rng: jax.Array) -> Array:
        # TD3 inference
        actions = self.state.actor.apply_fn(self.state.actor.params, observations, training=False, rng=rng)
        
        # DeterministicActor output is tanh -> [-1, 1].
        # Scale to action space
        actions = actions * self.config["action_scale"] + self.config["action_bias"]
        
        return actions
