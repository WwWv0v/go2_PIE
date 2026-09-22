"""Configuration dataclasses for PIE training."""

from dataclasses import dataclass

from mjlab.rl import RslRlPpoAlgorithmCfg


@dataclass
class PIEPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    auxiliary_loss_coef: float = 1.0
    velocity_loss_coef: float = 1.0
    foot_clearance_loss_coef: float = 1.0
    height_reconstruction_loss_coef: float = 1.0
    successor_loss_coef: float = 1.0
    kl_loss_coef: float = 4.0
    successor_target_group: str = "successor_target"
