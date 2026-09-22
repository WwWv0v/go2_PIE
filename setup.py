"""Installation script for Parkour MJLab training and deployment."""

from setuptools import find_packages, setup


setup(
    name="parkour_mjlab",
    version="0.1.0",
    packages=find_packages(include=("src", "src.*", "deploy", "deploy.*")),
    # Asset globs below are explicit; disabling automatic discovery prevents
    # setuptools from treating the XML/mesh directories as Python namespaces.
    include_package_data=False,
    package_data={
        "deploy.stair.sim2sim": [
            "assets/unitree_g1/LICENSE",
            "assets/unitree_g1/ASSET_SOURCE.md",
            "assets/unitree_g1/*.xml",
            "assets/unitree_g1/meshes/*.STL",
        ],
        "deploy.pie.sim2sim": ["assets/*.xml"],
        "src.assets.robots.unitree_go2": [
            "xmls/*.xml",
            "xmls/assets/*.obj",
        ],
    },
    install_requires=[
        "mjlab==1.2.0",
        "mujoco==3.5.0",
        "mujoco-warp==3.5.0",
        "numpy",
        "onnxruntime",
        "prettytable",
        "pygame>=2.6",
        "scipy",
        "torchrunx",
        "tyro",
        "wandb",
    ],
    extras_require={"test": ["pytest", "ruff"]},
)
