"""Procedural terrain set owned by the standalone PIE task."""

from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg


PIE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.10),
        "stairs_up": BoxPyramidStairsTerrainCfg(
            proportion=0.45,
            step_height_range=(0.03, 0.25),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.5,
        ),
        "stairs_down": BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.45,
            step_height_range=(0.03, 0.25),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.5,
        ),
    },
    add_lights=True,
)
