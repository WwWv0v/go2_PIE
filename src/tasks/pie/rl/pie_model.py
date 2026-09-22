"""RSL-RL actor implementing the PIE implicit-explicit estimator."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, resolve_nn_activation, unpad_trajectories
from tensordict import TensorDict


class PIEActorModel(nn.Module):
    
    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict[str, Any] | None = None,
        cnn_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(cnn_cfg or {})

        self.proprio_group = str(cfg.get("proprio_group", "actor"))
        self.history_group = str(cfg.get("history_group", "proprio_history"))
        self.depth_group = str(cfg.get("depth_group", "camera"))
        self.velocity_target_group = str(
            cfg.get("velocity_target_group", "velocity_target")
        )
        self.height_target_group = str(cfg.get("height_target_group", "height_target"))
        self.foot_target_group = str(
            cfg.get("foot_target_group", "foot_clearance_target")
        )
        self.successor_target_group = str(
            cfg.get("successor_target_group", "successor_target")
        )
        required_actor_groups = (
            self.proprio_group,
            self.history_group,
            self.depth_group,
        )
        missing = [
            name for name in required_actor_groups if name not in obs_groups[obs_set]
        ]
        if missing:
            raise ValueError(
                f"PIE actor groups {missing} are absent from obs_groups[{obs_set!r}]="
                f"{obs_groups[obs_set]}."
            )
        missing_obs = [
            name
            for name in (
                *required_actor_groups,
                self.velocity_target_group,
                self.height_target_group,
                self.foot_target_group,
                self.successor_target_group,
            )
            if name not in obs
        ]
        if missing_obs:
            raise ValueError(f"PIE observations are missing groups: {missing_obs}.")

        self.proprio_dim = int(obs[self.proprio_group].shape[-1])
        self.history_dim = int(obs[self.history_group].shape[-1])
        self.velocity_dim = int(obs[self.velocity_target_group].shape[-1])
        self.height_map_dim = int(obs[self.height_target_group].shape[-1])
        self.foot_dim = int(obs[self.foot_target_group].shape[-1])
        self.successor_dim = int(obs[self.successor_target_group].shape[-1])
        self.history_length = int(cfg.get("history_length", 10))
        if self.proprio_dim != 45:
            raise ValueError(
                f"PIE expects 45 current proprioceptive values, got {self.proprio_dim}."
            )
        if self.history_dim != self.proprio_dim * self.history_length:
            raise ValueError(
                f"PIE expects {self.history_length}x{self.proprio_dim} proprio history, "
                f"got {self.history_dim}."
            )

        configured_depth_shape = cfg.get("depth_shape", (2, 60, 86))
        self.depth_shape = tuple(int(v) for v in configured_depth_shape)
        self.depth_channels, self.depth_height, self.depth_width = self.depth_shape
        self.depth_flat_dim = self.depth_channels * self.depth_height * self.depth_width
        if int(obs[self.depth_group].shape[-1]) != self.depth_flat_dim:
            raise ValueError(
                f"PIE depth group has {obs[self.depth_group].shape[-1]} values; "
                f"expected {self.depth_flat_dim} from depth_shape={self.depth_shape}."
            )

        self.token_dim = int(cfg.get("token_dim", 64))
        self.map_latent_dim = int(cfg.get("map_latent_dim", 16))
        self.vae_latent_dim = int(cfg.get("vae_latent_dim", 16))
        self.memory_hidden_dim = int(cfg.get("memory_hidden_dim", 128))
        self.memory_num_layers = int(cfg.get("memory_num_layers", 1))
        self.obs_normalization = bool(obs_normalization)
        if self.obs_normalization:
            self.proprio_normalizer = EmpiricalNormalization(self.proprio_dim)
            self.history_normalizer = EmpiricalNormalization(self.history_dim)
        else:
            self.proprio_normalizer = nn.Identity()
            self.history_normalizer = nn.Identity()

        history_hidden = tuple(
            int(v) for v in cfg.get("history_hidden_dims", (256, 128))
        )
        self.history_encoder = MLP(
            self.history_dim,
            self.token_dim,
            history_hidden,
            activation,
        )
        self.depth_encoder = self._make_depth_encoder(cfg)
        self.depth_token_norm = nn.LayerNorm(self.token_dim)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(cfg.get("attention_heads", 1)),
            dim_feedforward=int(cfg.get("transformer_ff_dim", 256)),
            dropout=float(cfg.get("transformer_dropout", 0.0)),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.cross_modal_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=int(cfg.get("transformer_layers", 2)),
        )
        self.memory = nn.GRU(
            input_size=2 * self.token_dim,
            hidden_size=self.memory_hidden_dim,
            num_layers=self.memory_num_layers,
        )

        self.velocity_head = nn.Linear(self.memory_hidden_dim, self.velocity_dim)
        self.map_latent_head = nn.Linear(self.memory_hidden_dim, self.map_latent_dim)
        self.foot_clearance_head = nn.Linear(self.memory_hidden_dim, self.foot_dim)
        self.vae_mu_head = nn.Linear(self.memory_hidden_dim, self.vae_latent_dim)
        self.vae_logvar_head = nn.Linear(self.memory_hidden_dim, self.vae_latent_dim)

        estimator_output_dim = (
            self.velocity_dim
            + self.map_latent_dim
            + self.foot_dim
            + self.vae_latent_dim
        )
        self.actor_input_dim = self.proprio_dim + estimator_output_dim
        self.successor_decoder = MLP(
            estimator_output_dim,
            self.successor_dim,
            tuple(int(v) for v in cfg.get("successor_decoder_dims", (64, 128))),
            activation,
        )
        self.height_decoder = MLP(
            self.map_latent_dim,
            self.height_map_dim,
            tuple(int(v) for v in cfg.get("height_decoder_dims", (64, 128))),
            activation,
        )

        if distribution_cfg is not None:
            dist_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(
                dist_cfg.pop("class_name")
            )  # type: ignore[assignment]
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            actor_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            actor_output_dim = output_dim
        self.mlp = MLP(self.actor_input_dim, actor_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

        self.memory_hidden: torch.Tensor | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        estimates, _new_hidden = self._estimate_tensors(
            obs[self.proprio_group],
            obs[self.history_group],
            obs[self.depth_group],
            masks=masks,
            hidden_state=hidden_state,
            update_internal=masks is None and hidden_state is None,
            sample_latent=self.training,
        )
        actor_input = torch.cat(
            [self.proprio_normalizer(obs[self.proprio_group]), estimates["latent"]],
            dim=-1,
        )
        if masks is not None:
            actor_input = unpad_trajectories(actor_input, masks)
        actor_output = self.mlp(actor_input)
        if self.distribution is None:
            return actor_output
        if stochastic_output:
            self.distribution.update(actor_output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(actor_output)

    def auxiliary_losses(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> dict[str, torch.Tensor]:
        """Return the five PIE estimator losses before coefficient weighting."""
        estimates, _new_hidden = self._estimate_tensors(
            obs[self.proprio_group],
            obs[self.history_group],
            obs[self.depth_group],
            masks=masks,
            hidden_state=hidden_state,
            update_internal=False,
            sample_latent=True,
        )
        successor_hat = self.successor_decoder(estimates["latent"])
        height_hat = self.height_decoder(estimates["map_latent"])
        mu = estimates["mu"]
        logvar = estimates["logvar"]
        kl_per_sample = -0.5 * torch.mean(
            1.0 + logvar - mu.square() - logvar.exp(), dim=-1
        )
        return {
            "velocity": self._masked_mse(
                estimates["velocity"], obs[self.velocity_target_group], masks
            ),
            "foot_clearance": self._masked_mse(
                estimates["foot_clearance"], obs[self.foot_target_group], masks
            ),
            "height_reconstruction": self._masked_mse(
                height_hat, obs[self.height_target_group], masks
            ),
            "successor": self._masked_mse(
                successor_hat, obs[self.successor_target_group], masks
            ),
            "kl": self._masked_mean(kl_per_sample, masks),
        }

    def _estimate_tensors(
        self,
        proprio: torch.Tensor,
        history: torch.Tensor,
        depth: torch.Tensor,
        *,
        masks: torch.Tensor | None,
        hidden_state: torch.Tensor | None,
        update_internal: bool,
        sample_latent: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        history_token = self.history_encoder(self.history_normalizer(history))
        depth_tokens = self._encode_depth(depth)
        leading_shape = history_token.shape[:-1]
        tokens = torch.cat([history_token.unsqueeze(-2), depth_tokens], dim=-2)
        token_count = tokens.shape[-2]
        fused = self.cross_modal_transformer(
            tokens.reshape(-1, token_count, self.token_dim)
        )
        fused = fused.reshape(*leading_shape, token_count, self.token_dim)
        memory_input = torch.cat(
            [fused[..., 0, :], fused[..., 1:, :].mean(dim=-2)], dim=-1
        )

        if masks is not None:
            if hidden_state is None:
                raise ValueError(
                    "PIE recurrent update requires the rollout hidden state."
                )
            memory_output, new_hidden = self.memory(memory_input, hidden_state)
        else:
            memory_output, new_hidden = self.memory(
                memory_input.unsqueeze(0),
                hidden_state if hidden_state is not None else self.memory_hidden,
            )
            memory_output = memory_output.squeeze(0)
            if update_internal:
                self.memory_hidden = new_hidden

        velocity = self.velocity_head(memory_output)
        map_latent = self.map_latent_head(memory_output)
        foot_clearance = self.foot_clearance_head(memory_output)
        mu = self.vae_mu_head(memory_output)
        logvar = self.vae_logvar_head(memory_output).clamp(-10.0, 5.0)
        if sample_latent:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        latent = torch.cat([velocity, map_latent, foot_clearance, z], dim=-1)
        return {
            "velocity": velocity,
            "map_latent": map_latent,
            "foot_clearance": foot_clearance,
            "mu": mu,
            "logvar": logvar,
            "latent": latent,
        }, new_hidden

    def _make_depth_encoder(self, cfg: dict[str, Any]) -> nn.Sequential:
        channels = [int(v) for v in cfg.get("output_channels", (32, 64, 64))]
        channels[-1] = self.token_dim
        kernels = [int(v) for v in cfg.get("kernel_size", (8, 4, 3))]
        strides = [int(v) for v in cfg.get("stride", (4, 2, 1))]
        padding = cfg.get("padding", 0)
        layers: list[nn.Module] = []
        in_channels = self.depth_channels
        for index, out_channels in enumerate(channels):
            kernel = kernels[min(index, len(kernels) - 1)]
            stride = strides[min(index, len(strides) - 1)]
            if isinstance(padding, (tuple, list)):
                pad = int(padding[min(index, len(padding) - 1)])
            else:
                pad = int(padding)
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels, out_channels, kernel, stride=stride, padding=pad
                    ),
                    resolve_nn_activation(str(cfg.get("cnn_activation", "elu"))),
                ]
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        leading_shape = depth.shape[:-1]
        image = depth.reshape(-1, *self.depth_shape) - 0.5
        feature = self.depth_encoder(image)
        tokens = feature.flatten(start_dim=2).transpose(1, 2)
        tokens = self.depth_token_norm(tokens)
        return tokens.reshape(*leading_shape, tokens.shape[-2], self.token_dim)

    @staticmethod
    def _masked_mean(values: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
        if masks is None:
            return values.mean()
        mask = masks.to(dtype=values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @classmethod
    def _masked_mse(
        cls,
        prediction: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor | None,
    ) -> torch.Tensor:
        return cls._masked_mean((prediction - target).square().mean(dim=-1), masks)

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        if dones is None:
            self.memory_hidden = hidden_state
            return
        if self.memory_hidden is not None:
            self.memory_hidden[:, dones == 1, :] = 0.0

    def get_hidden_state(self):
        return self.memory_hidden

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones
        if self.memory_hidden is not None:
            self.memory_hidden = self.memory_hidden.detach()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.proprio_normalizer.update(obs[self.proprio_group])  # type: ignore[attr-defined]
            self.history_normalizer.update(obs[self.history_group])  # type: ignore[attr-defined]

    @property
    def output_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        assert self.distribution is not None
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _ExportPIEActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        return _ExportPIEActor(self)


class _ExportPIEActor(nn.Module):
    is_recurrent: bool = True

    def __init__(self, model: PIEActorModel) -> None:
        super().__init__()
        internal_hidden = model.memory_hidden
        distribution = model.distribution
        torch_distribution = getattr(distribution, "_distribution", None)
        model.memory_hidden = None
        if distribution is not None and hasattr(distribution, "_distribution"):
            distribution._distribution = None
        try:
            self.model = copy.deepcopy(model)
        finally:
            model.memory_hidden = internal_hidden
            if distribution is not None and hasattr(distribution, "_distribution"):
                distribution._distribution = torch_distribution
        self.model.eval()

    def forward(
        self,
        proprio: torch.Tensor,
        proprio_history: torch.Tensor,
        depth_history: torch.Tensor,
        memory_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        estimates, new_hidden = self.model._estimate_tensors(
            proprio,
            proprio_history,
            depth_history.flatten(start_dim=1),
            masks=None,
            hidden_state=memory_h,
            update_internal=False,
            sample_latent=False,
        )
        actor_input = torch.cat(
            [self.model.proprio_normalizer(proprio), estimates["latent"]], dim=-1
        )
        output = self.model.mlp(actor_input)
        if self.model.distribution is not None:
            output = self.model.distribution.deterministic_output(output)
        return output, new_hidden

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(1, self.model.proprio_dim),
            torch.zeros(1, self.model.history_dim),
            torch.zeros(1, *self.model.depth_shape),
            torch.zeros(self.model.memory_num_layers, 1, self.model.memory_hidden_dim),
        )

    @property
    def input_names(self) -> list[str]:
        return ["proprio", "proprio_history", "depth_history", "memory_h_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "memory_h_out"]
