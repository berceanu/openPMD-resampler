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

## Resampling

> The computer system for `GEANT4` simulation is made up of Intel Quard-core 2.66 GHz CPU and 12 GB DDR3 RAM and OS is Ubuntu 9.04 server version. It took about 3~4 hours to simulate with $10^7$ primary particles. [1]

We would like to implement various particle resampling strategies presented in the literature. [2]

## References

[1] Park, Seong Hee, et al. "A simulation for the optimization of bremsstrahlung radiation for nuclear applications using laser accelerated electron beam." Proceedings of FEL2010, Malmö, 2010. [Link](https://accelconf.web.cern.ch/FEL2010/papers/thpb13.pdf)

[2] Muraviev, A. et al. "Strategies for particle resampling in PIC simulations." Comput. Phys. Commun. 262, 107826 (2021). [Link](https://doi.org/10.1016/j.cpc.2021.107826)
