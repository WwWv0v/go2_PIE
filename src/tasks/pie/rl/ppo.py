"""PPO extension for the PIE estimator objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO


class PIEPPO(PPO):

    def __init__(
        self,
        *args,
        auxiliary_loss_coef: float = 1.0,
        velocity_loss_coef: float = 1.0,
        foot_clearance_loss_coef: float = 1.0,
        height_reconstruction_loss_coef: float = 1.0,
        successor_loss_coef: float = 1.0,
        kl_loss_coef: float = 4.0,
        successor_target_group: str = "successor_target",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.auxiliary_loss_coef = auxiliary_loss_coef
        self.auxiliary_coefficients = {
            "velocity": velocity_loss_coef,
            "foot_clearance": foot_clearance_loss_coef,
            "height_reconstruction": height_reconstruction_loss_coef,
            "successor": successor_loss_coef,
            "kl": kl_loss_coef,
        }
        self.successor_target_group = successor_target_group

    def act(self, obs):
        """Collect recurrent rollouts without retaining the rollout graph."""
        with torch.no_grad():
            self.transition.hidden_states = (
                self.actor.get_hidden_state(),
                self.critic.get_hidden_state(),
            )
            self.transition.actions = self.actor(obs, stochastic_output=True).detach()
            self.transition.values = self.critic(obs).detach()
            self.transition.actions_log_prob = self.actor.get_output_log_prob(
                self.transition.actions
            ).detach()
            self.transition.distribution_params = tuple(
                param.detach() for param in self.actor.output_distribution_params
            )
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        """Store the clean next proprioception as PIE's successor target."""
        if self.transition.observations is None:
            raise RuntimeError("PIEPPO.process_env_step() was called before act().")
        self.transition.observations[self.successor_target_group] = obs[
            self.successor_target_group
        ]
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_auxiliary = {name: 0.0 for name in self.auxiliary_coefficients}
        mean_rnd_loss = 0.0 if self.rnd else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for batch in generator:
            assert batch.observations is not None
            assert batch.actions is not None
            assert batch.advantages is not None
            assert batch.old_actions_log_prob is not None
            assert batch.old_distribution_params is not None
            assert batch.values is not None
            assert batch.returns is not None
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = tuple(
                param[:original_batch_size]
                for param in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean, op=torch.distributed.ReduceOp.SUM
                        )
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(
                actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
            )
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    (values - batch.returns).square(),
                    (value_clipped - batch.returns).square(),
                ).mean()
            else:
                value_loss = (batch.returns - values).square().mean()

            auxiliary_losses = self.actor.auxiliary_losses(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
            )
            weighted_auxiliary = sum(
                self.auxiliary_coefficients[name] * auxiliary_losses[name]
                for name in self.auxiliary_coefficients
            )
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.auxiliary_loss_coef * weighted_auxiliary
            )

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(
                        batch.observations[:original_batch_size]
                    )
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = nn.functional.mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                assert self.rnd_optimizer is not None
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            for name, value in auxiliary_losses.items():
                mean_auxiliary[name] += value.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_auxiliary = {
            name: value / num_updates for name, value in mean_auxiliary.items()
        }
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates

        self.storage.clear()
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            **{f"pie_{name}": value for name, value in mean_auxiliary.items()},
        }
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss
        return loss_dict
