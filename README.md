# openPMD-converter-Geant
Converter between openPMD and Geant

## Installation

```shell
$ pip install git+https://github.com/berceanu/openPMD-converter-Geant.git#egg=opmdtogeant
```

## Development

After installing [micromamba](https://mamba.readthedocs.io/en/latest/installation.html#micromamba), create a new environment via

```shell
$ micromamba create -n opmdgeant_env -f requirements.txt -c conda-forge
$ micromamba activate opmdgeant_env
```

Afterwards, clone the repo and run `python -m pip install -e .` in the same folder as this `README` file.

To start all over again, do 

```shell
$ micromamba deactivate
$ micromamba env remove -n opmdgeant_env`
```