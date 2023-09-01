# openPMD-Resampler
Resampling tools for openPMD PIC data

## Installation

```shell
$ pip install git+https://github.com/berceanu/openPMD-Resampler.git#egg=opmdresampler
```

## Usage

The main functionality of this project is demonstrated in the `usage.py` example script. This script provides a command-line interface for reading HDF5 files, visualizing phase space, resampling particles, and writing the results to a text file.

You can run the script with the following command:

```bash
python usage.py <path_to_your_hdf5_file>
```

[lwfa.h5](https://transfer.sequanium.de/qjhu1I2t56/lwfa.h5)

This will generate a log file named `output.md` in the current directory. This file contains the results of the script, including any logged messages and links to generated plots.

You can view the `usage.py` script [here](./usage.py), and an example `output.md` file [here](./output.md).


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

## Resampling

> The computer system for `GEANT4` simulation is made up of Intel Quard-core 2.66 GHz CPU and 12 GB DDR3 RAM and OS is Ubuntu 9.04 server version. It took about 3~4 hours to simulate with $10^7$ primary particles. [1]

We would like to implement various particle resampling strategies presented in the literature. [2]

## References

[1] Park, Seong Hee, et al. "A simulation for the optimization of bremsstrahlung radiation for nuclear applications using laser accelerated electron beam." Proceedings of FEL2010, Malmö, 2010. [Link](https://accelconf.web.cern.ch/FEL2010/papers/thpb13.pdf)

[2] Muraviev, A. et al. "Strategies for particle resampling in PIC simulations." Comput. Phys. Commun. 262, 107826 (2021). [Link](https://doi.org/10.1016/j.cpc.2021.107826)
