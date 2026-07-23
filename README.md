<img src="./resampler_logo.png" alt="Resampler Logo" width="400"/>


Resampling tools for `openPMD` PIC data.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17272699.svg)](https://doi.org/10.5281/zenodo.17272699)

## :bulb: Motivation

We often need to post-process the particle data from a PIC simulation, and pass it to additional tracking codes like [`GEANT`](#atom_symbol-geant4), [`GPT`](https://www.pulsar.nl/gpt/), [`SIMION`](https://simion.com) or [`Wake-T`](https://github.com/AngelFP/Wake-T). The original dataset can correspond to several **billion** particles, so one needs to reduce it to a manageable size, while conserving the main features of the underlying physics. This repository implements several resampling methods from the literature [[2]](#books-references), as well as a comprehensive suite of high-resolution [visualization](./plots/phase_space.png) tools, based on [Datashader](https://datashader.org).

## :rocket: Installation

We make use of the excellent [pixi](pixi.sh) package manager, which can be installed on Linux/macOS via

```console
$ curl -fsSL https://pixi.sh/install.sh | bash
```

One can then clone this repo via 

```console
$ git clone git@github.com:berceanu/openPMD-resampler.git
```

## :book: Usage

For an overview of the main functionality, see the [`usage.py`](./usage.py) example script and its [output](./output.md).
For production runs, use

```console
$ cd openPMD-resampler
$ pixi run start --opmd_path <path_to_your_openPMD_file> --species <particle_species_name> --mass <particle mass> --reduction_factor <k>
```

Replace descriptions between chevrons `<>` by relevant values, in this case the
path to the PIC output file, name of the particle species (`e_all` or
`e_highGamma` etc.), particle mass relative to the electron mass (default: 1.0) (1.0 for electron, 1836.152 for proton, or 22033.824 for carbon etc.), and an integer reduction factor `k`, typically between 2 and ~100.
If the initial PIC file has `N` macroparticles, the resulting reduced file will have `N/k`
macroparticles.

### Resampling algorithms

The resampling algorithm is selected via `--algorithm` (or `-a`):

- `thinning` (default): global leveling thinning [[2]](#books-references), controlled by `--reduction_factor`.
- `vranic`: particle merging of Vranic *et al.* [[4]](#books-references). Particles are binned into
  spatial and momentum cells, and each group of at least 4 particles sharing a cell is replaced by
  two macroparticles which exactly conserve total weight, momentum and energy.
- `voronoi`: Voronoi particle merging of Luu, Tückmantel and Pukhov [[5]](#books-references), as
  implemented in the [particle merger plugin](https://picongpu.readthedocs.io/en/0.5.0/usage/plugins/particleMerger.html)
  of PIConGPU. Particles are grouped into spatial cells which are subdivided recursively, first in
  position and then in momentum space, until the spread of every cluster falls below the given
  thresholds; each remaining cluster is replaced by a single macroparticle which exactly conserves
  total weight and momentum, with an energy error bounded by the momentum spread threshold.
- `upsample`: kernel *upsampling*, the inverse problem — a dataset with too few macroparticles for
  good post-processing statistics. Each parent macroparticle of weight `w` is split into `n`
  daughters of weight `w/n`, scattered around the parent by Gaussian kernel noise sized to the
  local phase-space spread (the weighted standard deviation of each coordinate within the parent's
  spatial cell, times a user bandwidth). The daughters are drawn in antithetic pairs (+δ, −δ), so
  total weight and momentum are conserved exactly per parent; energy is conserved to second order
  in the bandwidth (for massive particles an exact 1 → many split conserving all three is
  mathematically impossible), with the error logged.

**Which algorithm should I use?** Use `thinning` when you just want a target reduction factor `k`
and statistical conservation of the total charge is enough; it is the simplest option and the only
one that gives you the reduction factor directly, at the cost of extra sampling noise and the loss
of low-weight particles. Use `vranic` when every merge must exactly conserve weight, momentum *and*
energy; it gives the best physics fidelity, but the reduction factor is only controlled indirectly,
through the coarseness of the spatial and momentum bins. Use `voronoi` when you want adaptive
merging that follows the local phase-space structure, which works well for strongly non-uniform
beams; it conserves weight and momentum exactly, and energy only up to the momentum spread
threshold. Use `upsample` when you have the opposite problem — too *few* macroparticles (e.g. a
heavily reduced or filtered dataset) and the downstream analysis needs smoother statistics; it
multiplies the particle count by an exact factor while conserving weight and momentum exactly and
energy up to the kernel bandwidth. Note that upsampling adds no physical information — it smooths
the existing distribution for post-processing and is not a substitute for a better-resolved
simulation.

For example:

```console
$ pixi run start --opmd_path <path_to_your_openPMD_file> --species <particle_species_name> --mass <particle mass> --algorithm vranic --spatial_bins 16 16 16 --momentum_bins 16 16 16
```

The Vranic merging accepts the following options:

| Option | Default | Description |
| --- | --- | --- |
| `--spatial_bins NX NY NZ` | `16 16 16` | Number of position bins along x, y, z. |
| `--momentum_bins NP NTHETA NPHI` | `16 16 16` | Number of momentum bins. |
| `--momentum_coordinates {spherical,cartesian}` | `spherical` | Momentum space coordinates used for the binning; with `cartesian`, the bins are `NPX NPY NPZ`. |
| `--log_scale` | off | Bin the momentum norm logarithmically (spherical only), useful for broad energy spectra. |
| `--device D` | auto | PyTorch device the merge runs on, e.g. `cuda`, `cuda:1` or `cpu`. By default the GPU is used when one is available (NVIDIA CUDA and AMD ROCm builds of PyTorch both expose it as `cuda`), the CPU otherwise. |

The Voronoi merging accepts the following options, mirroring the PIConGPU plugin:

| Option | Default | Description |
| --- | --- | --- |
| `--spatial_bins NX NY NZ` | `16 16 16` | Number of initial Voronoi cells along x, y, z. |
| `--min_particles_to_merge N` | `8` | Minimum number of macroparticles in a Voronoi cell needed to merge them into a single macroparticle. |
| `--pos_spread_threshold T` | `0.5` | Below this spread in position a cell can be merged, in units of the initial spatial cell edge length. |
| `--abs_mom_spread_threshold T` | `-1` (disabled) | Below this absolute spread in momentum a cell can be merged, in units of $m_e c$. |
| `--rel_mom_spread_threshold T` | `-1` (disabled) | Below this spread in momentum relative to the cell's mean momentum a cell can be merged. Exactly one of the two momentum thresholds must be enabled. |
| `--min_mean_energy E` | `511.0` | Minimum mean kinetic energy in keV of a Voronoi cell needed to merge it (`0` disables the criterion). |
| `--device D` | auto | Same as for the Vranic merging. |

The default `--min_mean_energy` of 511 keV (~one electron rest mass) means clusters whose mean
kinetic energy falls below that threshold are silently left unmerged, so low-energy particles pass
through the resampling unreduced; set it to `0` to disable the criterion and merge regardless of
energy.

For example:

```console
$ pixi run start --opmd_path <path_to_your_openPMD_file> --species <particle_species_name> --mass <particle mass> --algorithm voronoi --rel_mom_spread_threshold 0.1 --spatial_bins 128 128 1024 --min_particles_to_merge 10
```

The kernel upsampling accepts the following options:

| Option | Default | Description |
| --- | --- | --- |
| `--upsampling_factor N` | `10` | Number of daughters per parent macroparticle; the output has exactly `N` times the input rows, each daughter carrying `1/N` of the parent's weight. |
| `--spatial_bins NX NY NZ` | `16 16 16` | Number of position bins along x, y, z used to estimate the local weighted phase-space spread. |
| `--position_bandwidth B` | `0.1` | Kernel width for the position coordinates, as a fraction of the local weighted standard deviation; `0` leaves positions unperturbed. |
| `--momentum_bandwidth B` | `0.1` | Same for the momentum coordinates; larger values smooth the momentum distribution more, at the cost of a larger (always positive) energy error. |
| `--device D` | auto | Same as for the Vranic merging. |

For example:

```console
$ pixi run start --opmd_path <path_to_your_openPMD_file> --species <particle_species_name> --mass <particle mass> --algorithm upsample --upsampling_factor 20 --momentum_bandwidth 0.05
```

Unlike thinning, the merging algorithms do not set the reduction factor directly: coarser binning
(fewer bins) or larger spread thresholds merge more aggressively, at the cost of more phase-space
smearing, while finer settings preserve the distribution better but merge less. Since the merged
(and upsampled) macroparticles have non-uniform weights, the `weights` column of the output file
must be taken into account by the tracking code. For photons, use `--mass 0.0`.

The merging and upsampling algorithms run on PyTorch. The default environment ships the CPU build, so no CUDA
packages are downloaded; to run on an NVIDIA GPU, use the `cuda` environment instead — it is the
only one that requires a CUDA driver:

```console
$ pixi run -e cuda start --opmd_path <path_to_your_openPMD_file> --algorithm vranic ...
```

Spatial cells never interact, so datasets larger than the GPU memory are automatically processed
in chunks of whole cells sized to the free GPU memory, without changing the result.

If you need a sample PIC output file for testing, you can download [lwfa.h5](https://transfer.sequanium.de/qjhu1I2t56/lwfa.h5) [212M].

The code works with `openPMD`-compatible PIC codes, such as [`WarpX`](https://github.com/ECP-WarpX/WarpX), [`PIConGPU`](https://github.com/ComputationalRadiationPhysics/picongpu), [`fbpic`](https://github.com/fbpic/fbpic), etc.

The runtime is typically a few minutes and the memory footprint is about twice the size of the input file.

The output is a CSV text file, of the following form:

```
position_x_m (m), position_y_m (m), position_z_m (m), momentum_x_mev_c (MeV/c), momentum_y_mev_c (MeV/c), momentum_z_mev_c (MeV/c), weights (1)
1.1201412540356980e-05,8.0062201241442832e-07,3.9551004545608885e-03,-9.1752357482910156e+00,-1.4616233825683594e+01,2.9899465942382812e+02,1.0000000000000000e+00
...
```

Positions are exported in meters and momenta in MeV/c, with the macroparticle `weights` as the last column.

With `--fortran_unformatted`, the output is instead a Fortran unformatted (sequential, record-based) binary file with the records `n (int32), x, y, z, ux, uy, uz, w (float32 arrays)`, where the momenta are written as normalized momentum u = p/(m·c) (dimensionless, openPMD-viewer's `ux` convention) rather than MeV/c. For massless species (`--mass 0.0`), the momenta are normalized by the electron mass instead, u = p/(m<sub>e</sub>·c), the usual PIC convention for photons. Note that this record format caps the output at ~1.07 billion particles (4 GiB per column record).

## :wrench: Development

All project dependencies are listed under [`pixi.toml`](pixi.toml).
Just create a fork, follow the install instructions and start making PRs.

Run the test suite with `pixi run test` (CPU), or `pixi run -e test-cuda test` to also exercise
the GPU code paths.

## :atom_symbol: GEANT4

For a computational estimate, here is a quote from Ref. [1]:

> The computer system for [`GEANT4`](https://geant4.web.cern.ch) simulation is made up of Intel Quad-core 2.66 GHz CPU and 12 GB DDR3 RAM and OS is Ubuntu 9.04 server version. It took about 3~4 hours to simulate with $10^7$ primary particles.

## :writing_hand: Citing

If you use `openPMD-resampler` in your research, please cite it as follows:

```bibtex
@software{berceanu_2025_17272699,
  author       = {Berceanu, Andrei},
  title        = {openPMD-resampler},
  month        = oct,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.5.1},
  doi          = {10.5281/zenodo.17272699},
  url          = {https://doi.org/10.5281/zenodo.17272699}
}
```

## :books: References

[1] Park, Seong Hee, et al. "A simulation for the optimization of bremsstrahlung radiation for nuclear applications using laser accelerated electron beam." Proceedings of FEL2010, Malmö, 2010. [PDF](https://accelconf.web.cern.ch/FEL2010/papers/thpb13.pdf)

[2] Muraviev, A. et al. "Strategies for particle resampling in PIC simulations." Comput. Phys. Commun. 262, 107826 (2021). [DOI](https://doi.org/10.1016/j.cpc.2021.107826)

[3] Shimazaki, Hideaki, and Shigeru Shinomoto. "A method for selecting the bin size of a time histogram." Neural computation 19.6, 1503 (2007). [DOI](https://doi.org/10.1162/neco.2007.19.6.1503)

[4] Vranic, Marija, et al. "Merging of macro-particles in particle-in-cell simulations." Comput. Phys. Commun. 191, 65 (2015). [DOI](https://doi.org/10.1016/j.cpc.2015.01.020)

[5] Luu, Phuc T., Tückmantel, T. and Pukhov, A. "Voronoi particle merging algorithm for PIC codes." Comput. Phys. Commun. 202, 165 (2016). [DOI](https://doi.org/10.1016/j.cpc.2016.01.009)

## :loudspeaker: Acknowledgements

We would like to acknowledge useful discussions with [Richard Pausch (HZDR)](https://github.com/PrometheusPi).

## :link: Similar Projects

- [Particle Reduction](https://github.com/ComputationalRadiationPhysics/particle_reduction)
- [Hi-Chi framework](https://github.com/hi-chi/pyHiChi)
