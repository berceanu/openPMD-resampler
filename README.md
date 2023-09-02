# :electron: openPMD-Resampler
Resampling tools for `openPMD` PIC data

## :bulb: Motivation

We often need to post-process the particle data from a PIC simulation, and pass it to additional tracking codes like [`GEANT`](#atom_symbol-geant4), [`GPT`](https://www.pulsar.nl/gpt/), [`SIMION`](https://simion.com) or [`Wake-T`](https://github.com/AngelFP/Wake-T). The original dataset can correspond to several **billion** particles, so one needs to reduce it to a manageable size, while conserving the main features of the underlying physics. This repository implements several resampling methods from the literature [[2]](#books-references), as well as a comprehensive suite of high-resolution [visualization](./plots/phase_space.png) tools.

## :rocket: Installation

```console
$ pip install git+https://github.com/berceanu/openPMD-Resampler.git#egg=opmdresampler
```

## :book: Usage

For an overview of the main functionality, see the [`usage.py`](./usage.py) example script and its [output](./output.md):

```console
$ python usage.py <path_to_your_openPMD_file>
```

Replace descriptions between chevrons `<>` by relevant values, in this case the path to the PIC output file.
If you need a sample PIC output file for testing, you can download [lwfa.h5](https://transfer.sequanium.de/qjhu1I2t56/lwfa.h5) [212M].

The code works with `openPMD`-compatible PIC codes, such as `WarpX`, `PIConGPU`, `fbpic`, etc.

The runtime is under one minute on an *M1 MacBook Air* and the memory footprint is ~1 GB RAM.

The output is a CSV text file, of the following form:

```
position_x_um (μm), position_y_um (μm), position_z_um (μm), momentum_x_mev_c (MeV/c), momentum_y_mev_c (MeV/c), momentum_z_mev_c (MeV/c)
1.1201412540356980e+01,8.0062201241442832e-01,3.9551004545608885e+03,-9.1752357482910156e+00,-1.4616233825683594e+01,2.9899465942382812e+02
...
```

## :wrench: Development

All project dependencies are listed under [`requirements.txt`](requirements.txt).

After installing your favorite `Python` distribution, for example [`micromamba`](https://mamba.readthedocs.io/en/latest/micromamba-installation.html#umamba-install), create a new environment:

```console
$ micromamba env create -f environment.yml
$ micromamba activate opmdresamplerenv
```

Clone the repo and run: 

```console
$ python -m pip install -e .
```

To start from scratch:

```console
$ micromamba deactivate
$ micromamba env remove -n opmdresamplerenv
```

## :atom_symbol: GEANT4

For a computational estimate, here is a quote from Ref. [1]:

> The computer system for `GEANT4` simulation is made up of Intel Quad-core 2.66 GHz CPU and 12 GB DDR3 RAM and OS is Ubuntu 9.04 server version. It took about 3~4 hours to simulate with $10^7$ primary particles.


## :books: References

[1] Park, Seong Hee, et al. "A simulation for the optimization of bremsstrahlung radiation for nuclear applications using laser accelerated electron beam." Proceedings of FEL2010, Malmö, 2010. [PDF](https://accelconf.web.cern.ch/FEL2010/papers/thpb13.pdf)

[2] Muraviev, A. et al. "Strategies for particle resampling in PIC simulations." Comput. Phys. Commun. 262, 107826 (2021). [DOI](https://doi.org/10.1016/j.cpc.2021.107826)

## :loudspeaker: Acknowledgements

We would like to acknowledge useful discussions with [Richard Pausch (HZDR)](https://github.com/PrometheusPi).

## :link: Similar Projects

- [Particle Reduction](https://github.com/ComputationalRadiationPhysics/particle_reduction)
- [Hi-Chi framework](https://github.com/hi-chi/pyHiChi)
