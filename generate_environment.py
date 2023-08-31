"""
This module is used to generate an environment file for a conda environment.
It reads the requirements from a text file and writes them into a yaml file.
The yaml file can then be used to create a conda environment with the specified dependencies.
"""
import yaml

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

env = {
    "name": "thinningenv",
    "channels": ["conda-forge"],
    "dependencies": requirements,
}

with open("environment.yml", "w", encoding="utf-8") as f:
    yaml.dump(env, f, sort_keys=False)
