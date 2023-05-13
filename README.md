# openPMD-converter-Geant
Converter between openPMD and Geant

## Development

After installing [micromamba](https://mamba.readthedocs.io/en/latest/installation.html#micromamba), create a new environment via

```shell
$ micromamba create -n opmdgeant_env -f requirements.txt -c conda-forge
$ micromamba activate opmdgeant_env
```

Afterwards, clone the repo and run `pip install -e .` in the same folder as this `README` file.

To start all over again, do `micromamba env remove -n opmdgeant_env`.