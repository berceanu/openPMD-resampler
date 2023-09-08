"""
This module provides a command-line interface for reading OpenPMD files, 
visualizing phase space, resampling particles, and writing the results to a text file.
"""
import argparse
from pathlib import Path

from opmdresampler.df_to_txt import DataFrameToFile
from opmdresampler.reader import ParticleDataReader
from opmdresampler.resampling import ParticleResampler
from opmdresampler.visualize_phase_space import PhaseSpaceVisualizer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("opmd_path", type=str, help="Path to the OpenPMD file")
    args = parser.parse_args()
    opmd_path = Path(args.opmd_path)
    ###############################

    # Create the dataframe
    df = ParticleDataReader.from_file(
        opmd_path, particle_species_name="e_highGamma"
    )  # or "e_all"

    # Create the phase space plots
    phase_space = PhaseSpaceVisualizer(df)
    phase_space.savefig("plots/phase_space.png")

    # Apply thinning algorithm to df, resulting in df_thin
    resampler = ParticleResampler(df)
    df_thin = resampler.global_leveling_thinning().set_weights_to(1).finalize()

    # Visualize both dataframes in order to see effects of thining
    phase_space_thin = PhaseSpaceVisualizer(df_thin)
    phase_space += phase_space_thin
    phase_space.savefig("plots/comparative_phase_space.png")

    # Write the reduced dataframe to a file
    DataFrameToFile(df_thin).exclude_weights().exclude_energy().write_to_file(
        opmd_path.with_suffix(".txt")
    )


if __name__ == "__main__":
    main()
