# openPMD-Resampler
Resampling tools for openPMD PIC data

## Motivation

We often need to post-process the particle data from a PIC code, and pass it to additional tracking codes like [`GEANT`](#geant4), `GPT` or `SIMION`. The original dataset can correspond to several billion particles, so one needs to reduce it to a manageable size, while conserving the main features of the underlying physics. This repository implements several resampling methods from the literature [2], as well as a comprehensive suite of high-resolution visualization tools.

## Installation

```shell
pip install git+https://github.com/berceanu/openPMD-Resampler.git#egg=opmdresampler
```

## Usage

For an overview of the functionality, see the [`usage.py`](./usage.py) example script and its [output](./output.md):

```bash
python usage.py <path_to_your_pic_output_file>
```

If you need a sample PIC output file, you can download [lwfa.h5](https://transfer.sequanium.de/qjhu1I2t56/lwfa.h5) [212M].

## Development

All project dependencies are listed in the file [`requirements.txt`](requirements.txt).
`setup.py` reads them from there, and so does the `conda` environment file `environment.yml`.

After installing any `conda` Python distribution, for example [`micromamba`](https://mamba.readthedocs.io/en/latest/installation.html#micromamba), create a new environment: 

```shell
micromamba env create -f environment.yml
micromamba activate opmdresamplerenv
```

Clone the repository and run 

```shell
python -m pip install -e .
```

To start from scratch, do 

```shell
micromamba deactivate
micromamba env remove -n opmdresamplerenv
```

## GEANT4

For a computational estimate, here is a quote from Ref. [1]:

> The computer system for `GEANT4` simulation is made up of Intel Quard-core 2.66 GHz CPU and 12 GB DDR3 RAM and OS is Ubuntu 9.04 server version. It took about 3~4 hours to simulate with $10^7$ primary particles.


## References

[1] Park, Seong Hee, et al. "A simulation for the optimization of bremsstrahlung radiation for nuclear applications using laser accelerated electron beam." Proceedings of FEL2010, Malmö, 2010. [Link](https://accelconf.web.cern.ch/FEL2010/papers/thpb13.pdf)

[2] Muraviev, A. et al. "Strategies for particle resampling in PIC simulations." Comput. Phys. Commun. 262, 107826 (2021). [Link](https://doi.org/10.1016/j.cpc.2021.107826)
