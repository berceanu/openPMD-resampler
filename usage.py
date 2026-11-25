"""
This module provides a command-line interface for reading OpenPMD files,
visualizing phase space, resampling particles, and writing the results to a text file.
"""
import argparse
from pathlib import Path

from openpmd_resampler.calorimeter import particles_from_calorimeter
from openpmd_resampler.df_to_txt import DataFrameToFile
from openpmd_resampler.reader import ParticleDataReader
from openpmd_resampler.resampling import ParticleResampler
from openpmd_resampler.units import constants
from openpmd_resampler.visualize_phase_space import PhaseSpaceVisualizer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--opmd_path", type=str, help="Path to the OpenPMD particle file")
    source.add_argument("--calorimeter_path", type=str,
                        help="Path to a PIConGPU particle calorimeter file (e.g. H_calorimeter_forward_80000.bp5);"
                             " macroparticles are generated from it instead of read from --opmd_path")
    parser.add_argument("--density_path", type=str, default=None,
                        help="Calorimeter input: openPMD file with the species density mesh at the same iteration"
                             " (e.g. simOutput/openPMD/simData_080000.bp5), used to draw positions;"
                             " omit to place all particles at the origin")
    parser.add_argument("--density_mesh", type=str, default=None,
                        help="Calorimeter input: name of the density mesh (default: <species>_density, e.g. H_all_density)")
    parser.add_argument("--calorimeter_sampling", type=str, default="random", choices=["random", "per_bin"],
                        help="Calorimeter input: 'random' draws --num_macroparticles equal-weight macroparticles;"
                             " 'per_bin' puts --particles_per_bin variable-weight macroparticles in every non-empty bin"
                             " (default: random)")
    parser.add_argument("--num_macroparticles", type=float, default=1e6,
                        help="Calorimeter input, random sampling: number of macroparticles (default: 1e6)")
    parser.add_argument("--particles_per_bin", type=int, default=1,
                        help="Calorimeter input, per_bin sampling: macroparticles per non-empty bin (default: 1)")
    parser.add_argument("--calorimeter_energy_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="Calorimeter input: only use energy bins entirely inside [MIN, MAX] MeV (default: all)")
    parser.add_argument("--species", "-s", type=str, default="e_all",
                        help="Particle species name (default: e_all)")
    parser.add_argument("--mass", "-m", type=float, default=1.0,
                        help="Particle mass relative to the electron mass (default: 1.0)")
    parser.add_argument("--algorithm", "-a", type=str, default=None,
                        choices=["thinning", "vranic", "voronoi", "upsample", "none"],
                        help="Resampling algorithm: global leveling thinning, Vranic merging, Voronoi merging, kernel upsampling"
                             " or none (default: thinning for --opmd_path, none for --calorimeter_path)")
    parser.add_argument("--reduction_factor", "-k", type=float, default=2.0,
                        help="The 'k' level for global leveling thinning (default: 2.0)")
    parser.add_argument("--upsampling_factor", type=int, default=10,
                        help="Kernel upsampling: number of daughters each macroparticle is split into (default: 10)")
    parser.add_argument("--position_bandwidth", type=float, default=0.1,
                        help="Kernel upsampling: position kernel width as a fraction of the local weighted spread; 0 disables (default: 0.1)")
    parser.add_argument("--momentum_bandwidth", type=float, default=0.1,
                        help="Kernel upsampling: momentum kernel width as a fraction of the local weighted spread; 0 disables (default: 0.1)")
    parser.add_argument("--spatial_bins", type=int, nargs=3, default=[16, 16, 16],
                        metavar=("NX", "NY", "NZ"),
                        help="Vranic/Voronoi merging: number of spatial bins (default: 16 16 16)")
    parser.add_argument("--momentum_bins", type=int, nargs=3, default=[16, 16, 16],
                        metavar=("NP", "NTHETA", "NPHI"),
                        help="Vranic merging: number of momentum bins (default: 16 16 16)")
    parser.add_argument("--momentum_coordinates", type=str, default="spherical",
                        choices=["spherical", "cartesian"],
                        help="Vranic merging: momentum space coordinates (default: spherical)")
    parser.add_argument("--log_scale", action="store_true",
                        help="Vranic merging: bin the momentum norm logarithmically.")
    parser.add_argument("--device", type=str, default=None,
                        help="Vranic/Voronoi merging: PyTorch device, e.g. 'cuda', 'cuda:1' or 'cpu'"
                             " (default: the GPU if available, both NVIDIA CUDA and AMD ROCm, else the CPU)")
    parser.add_argument("--min_particles_to_merge", type=int, default=8,
                        help="Voronoi merging: minimum number of macroparticles in a Voronoi cell needed to merge them (default: 8)")
    parser.add_argument("--pos_spread_threshold", type=float, default=0.5,
                        help="Voronoi merging: below this spread in position a cell can be merged, in units of the initial spatial cell edge length (default: 0.5)")
    parser.add_argument("--abs_mom_spread_threshold", type=float, default=-1.0,
                        help="Voronoi merging: below this absolute spread in momentum a cell can be merged, in units of m_e*c; disabled for -1 (default)")
    parser.add_argument("--rel_mom_spread_threshold", type=float, default=-1.0,
                        help="Voronoi merging: below this spread in momentum relative to the mean momentum a cell can be merged; disabled for -1 (default). Exactly one momentum spread threshold must be enabled.")
    parser.add_argument("--min_mean_energy", type=float, default=511.0,
                        help="Voronoi merging: minimum mean kinetic energy in keV of a Voronoi cell needed to merge it (default: 511.0)")
    parser.add_argument("--no_plot", action="store_true",
                        help="If set, the phase space plot will not be created.")
    parser.add_argument("--fortran_unformatted", action="store_true",
                        help="If set, write output as a Fortran unformatted binary file instead of CSV,"
                             " with momenta as normalized momentum u = p/(m*c) instead of MeV/c.")
    parser.add_argument("--no_csv", action="store_true",
                        help="If set, the resulting dataframe will not be saved to file.")

    args = parser.parse_args()
    particle_species_name = args.species
    particle_species_mass = args.mass
    reduction_factor = args.reduction_factor
    fortran_unformatted = args.fortran_unformatted

    # Create the dataframe, either from PIC particles or from a calorimeter
    if args.calorimeter_path is not None:
        input_path = Path(args.calorimeter_path)
        df = particles_from_calorimeter(
            input_path,
            density_path=args.density_path,
            density_mesh=args.density_mesh or f"{particle_species_name}_density",
            mass_mev_c2=particle_species_mass * constants.electron_mass_mev_c2,
            mode=args.calorimeter_sampling,
            num_macroparticles=int(args.num_macroparticles),
            particles_per_bin=args.particles_per_bin,
            energy_range_mev=args.calorimeter_energy_range,
        )
        algorithm = args.algorithm or "none"
    else:
        input_path = Path(args.opmd_path)
        df = ParticleDataReader.from_file(input_path, particle_species_name=particle_species_name, particle_species_mass=particle_species_mass)
        algorithm = args.algorithm or "thinning"

    # Apply the resampling algorithm to df, resulting in df_thin
    resampler = ParticleResampler(df, particle_species_mass=particle_species_mass)
    if algorithm == "none":
        df_thin = resampler.finalize()
    elif algorithm == "vranic":
        df_thin = resampler.vranic_merging(
            spatial_bins=tuple(args.spatial_bins),
            momentum_bins=tuple(args.momentum_bins),
            momentum_coordinates=args.momentum_coordinates,
            log_scale=args.log_scale,
            device=args.device,
        ).finalize()
    elif algorithm == "voronoi":
        df_thin = resampler.voronoi_merging(
            spatial_bins=tuple(args.spatial_bins),
            min_particles_to_merge=args.min_particles_to_merge,
            pos_spread_threshold=args.pos_spread_threshold,
            abs_mom_spread_threshold=args.abs_mom_spread_threshold,
            rel_mom_spread_threshold=args.rel_mom_spread_threshold,
            min_mean_energy_kev=args.min_mean_energy,
            device=args.device,
        ).finalize()
    elif algorithm == "upsample":
        # Daughter weights are w/n, non-uniform in general, so no set_weights_to(1).
        df_thin = resampler.kernel_upsampling(
            upsampling_factor=args.upsampling_factor,
            spatial_bins=tuple(args.spatial_bins),
            position_bandwidth=args.position_bandwidth,
            momentum_bandwidth=args.momentum_bandwidth,
            device=args.device,
        ).finalize()
    else:
        df_thin = resampler.global_leveling_thinning(k=reduction_factor).finalize()

    # Write the reduced dataframe to a file
    suffix = ".dat"
    writer = DataFrameToFile(df_thin).exclude_energy()
    if fortran_unformatted:
        # Fortran consumers expect normalized momentum u = p/(m*c).
        writer.momentum_in_mc(particle_species_mass * constants.electron_mass_mev_c2)
    writer.write_to_file(input_path.with_suffix(suffix), fortran_unformatted=fortran_unformatted)

    # Visualize both dataframes in order to see effects of thining
    phase_space = PhaseSpaceVisualizer(df, label="PIC data")
    phase_space_thin = PhaseSpaceVisualizer(df_thin, label="Resampled data")
    comparative_phase_space = phase_space + phase_space_thin

    comparative_phase_space.create_plot().savefig(f"{input_path.parent}/comparative_phase_space_{algorithm}.png")


if __name__ == "__main__":
    main()
