"""
This module provides a command-line interface for reading OpenPMD files,
visualizing phase space, resampling particles, and writing the results to a text file.
"""
import argparse
from pathlib import Path

from openpmd_resampler.df_to_txt import DataFrameToFile
from openpmd_resampler.reader import ParticleDataReader
from openpmd_resampler.resampling import ParticleResampler
from openpmd_resampler.visualize_phase_space import PhaseSpaceVisualizer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--opmd_path", type=str, help="Path to the OpenPMD file")
    parser.add_argument("--species", "-s", type=str, default="e_all",
                        help="Particle species name (default: e_all)")
    parser.add_argument("--mass", "-m", type=float, default=1.0,
                        help="Particle mass relative to the electron mass (default: 1.0)")
    parser.add_argument("--reduction_factor", "-k", type=float, default=2.0,
                        help="The 'k' level for global leveling thinning (default: 2.0)")
    parser.add_argument("--no_plot", action="store_true",
                        help="If set, the phase space plot will not be created.")
    parser.add_argument("--no_csv", action="store_true",
                        help="If set, the resulting dataframe will not be saved to file.")

    args = parser.parse_args()
    opmd_path = Path(args.opmd_path)
    particle_species_name = args.species
    particle_species_mass = args.mass
    reduction_factor = args.reduction_factor

    # Create the dataframe
    df = ParticleDataReader.from_file(opmd_path, particle_species_name=particle_species_name, particle_species_mass=particle_species_mass)

    # Apply thinning algorithm to df, resulting in df_thin
    resampler = ParticleResampler(df)
    df_thin = resampler.global_leveling_thinning(k=reduction_factor).set_weights_to(1).finalize()

    # Visualize both dataframes in order to see effects of thining
    phase_space = PhaseSpaceVisualizer(df, label="PIC data")
    phase_space_thin = PhaseSpaceVisualizer(df_thin, label="Resampled data")
    comparative_phase_space = phase_space + phase_space_thin
    comparative_phase_space.create_plot().savefig("plots/comparative_phase_space.png")

    # Write the reduced dataframe to a file
    DataFrameToFile(df_thin).exclude_weights().exclude_energy().write_to_file(
        opmd_path.with_suffix(".dat")
    )


if __name__ == "__main__":
    main()
