"""Standalone RSL-RL configuration for Unitree Go2 PIE."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from src.tasks.pie.rl import PIEPpoAlgorithmCfg


_PIE_ACTOR_CLS = "src.tasks.pie.rl.pie_model:PIEActorModel"
_PIE_PPO_CLS = "src.tasks.pie.rl.ppo:PIEPPO"
_PIE_ACTOR_CFG = {
    "proprio_group": "actor",
    "history_group": "proprio_history",
    "depth_group": "camera",
    "velocity_target_group": "velocity_target",
    "height_target_group": "height_target",
    "foot_target_group": "foot_clearance_target",
    "successor_target_group": "successor_target",
    "history_length": 10,
    "depth_shape": (2, 60, 86),
    "token_dim": 64,
    "attention_heads": 1,
    "transformer_layers": 2,
    "transformer_ff_dim": 256,
    "transformer_dropout": 0.0,
    "memory_hidden_dim": 128,
    "memory_num_layers": 1,
    "map_latent_dim": 16,
    "vae_latent_dim": 16,
    "history_hidden_dims": (256, 128),
    "successor_decoder_dims": (64, 128),
    "height_decoder_dims": (64, 128),
    "output_channels": (32, 64, 64),
    "kernel_size": (8, 4, 3),
    "stride": (4, 2, 1),
    "padding": 0,
    "cnn_activation": "elu",
}


def unitree_go2_pie_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create the policy, estimator, and PPO settings for Go2 PIE."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name=_PIE_ACTOR_CLS,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_PIE_ACTOR_CFG,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=PIEPpoAlgorithmCfg(
            class_name=_PIE_PPO_CLS,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            auxiliary_loss_coef=1.0,
            velocity_loss_coef=1.0,
            foot_clearance_loss_coef=1.0,
            height_reconstruction_loss_coef=1.0,
            successor_loss_coef=1.0,
            kl_loss_coef=4.0,
        ),
        obs_groups={
            "actor": ("actor", "proprio_history", "camera"),
            "critic": ("critic",),
        },
        experiment_name="go2_pie",
        save_interval=1000,
        num_steps_per_env=24,
        max_iterations=20000,
    )
