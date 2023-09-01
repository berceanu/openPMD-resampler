# :electron: openPMD-Resampler
Resampling tools for `openPMD` PIC data

## :bulb: Motivation

We often need to post-process the particle data from a PIC simulation, and pass it to additional tracking codes like [`GEANT`](#atom_symbol-geant4), [`GPT`](https://www.pulsar.nl/gpt/), [`SIMION`](https://simion.com) or [`Wake-T`](https://github.com/AngelFP/Wake-T). The original dataset can correspond to several **billion** particles, so one needs to reduce it to a manageable size, while conserving the main features of the underlying physics. This repository implements several resampling methods from the literature [[2]](#books-references), as well as a comprehensive suite of high-resolution visualization tools.

## :rocket: Installation

```shell
pip install git+https://github.com/berceanu/openPMD-Resampler.git#egg=opmdresampler
```

## :book: Usage

For an overview of the main functionality, see the [`usage.py`](./usage.py) example script and its [output](./output.md):

```bash
python usage.py <path_to_your_pic_output_file>
```

If you need a sample PIC output file, you can download [lwfa.h5](https://transfer.sequanium.de/qjhu1I2t56/lwfa.h5) [212M].

The runtime is under one minute on an *M1 MacBook Air* and the memory footprint is ~1 GB RAM.

## :wrench: Development

All project dependencies are listed under [`requirements.txt`](requirements.txt).

After installing your favorite `Python` distribution, for example [`micromamba`](https://mamba.readthedocs.io/en/latest/micromamba-installation.html#umamba-install), create a new environment:

```shell
micromamba env create -f environment.yml
micromamba activate opmdresamplerenv
```

Clone the repo and run: 

```shell
python -m pip install -e .
```

To start from scratch:

```shell
micromamba deactivate
micromamba env remove -n opmdresamplerenv
```

## :atom_symbol: GEANT4

For a computational estimate, here is a quote from Ref. [1]:

> The computer system for `GEANT4` simulation is made up of Intel Quad-core 2.66 GHz CPU and 12 GB DDR3 RAM and OS is Ubuntu 9.04 server version. It took about 3~4 hours to simulate with $10^7$ primary particles.


## :books: References

[1] Park, Seong Hee, et al. "A simulation for the optimization of bremsstrahlung radiation for nuclear applications using laser accelerated electron beam." Proceedings of FEL2010, Malmö, 2010. [PDF](https://accelconf.web.cern.ch/FEL2010/papers/thpb13.pdf)

[2] Muraviev, A. et al. "Strategies for particle resampling in PIC simulations." Comput. Phys. Commun. 262, 107826 (2021). [DOI](https://doi.org/10.1016/j.cpc.2021.107826)

## :loudspeaker: Acknowledgements

We would like to acknowledge useful discussions with Dr. Richard Pausch.

## :link: Similar Projects

- [Particle Reduction](https://github.com/ComputationalRadiationPhysics/particle_reduction)
