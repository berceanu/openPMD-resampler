"""
This module provides a command-line interface for reading OpenPMD files, 
visualizing phase space, resampling particles, and writing the results to a text file.
"""
import argparse
import sys
from pathlib import Path

from opmdresampler.df_to_txt import DataFrameToFile
from opmdresampler.log import logger
from opmdresampler.reader import ParticleDataReader
from opmdresampler.resampling import ParticleResampler
from opmdresampler.utils import dataset_info
from opmdresampler.visualize_phase_space import PhaseSpaceVisualizer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("opmd_path", type=str, help="Path to the OpenPMD file")
    args = parser.parse_args()
    opmd_path = Path(args.opmd_path)

    # Log the command used to run the script
    command = " ".join(sys.argv)
    logger.info("## Output\n")
    logger.info("This is the output of `%s`\n", command)

    # Create the dataframe
    df = ParticleDataReader.from_file(opmd_path, particle_species_name="e_highGamma")

    # Create the phase space plots
    phase_space = PhaseSpaceVisualizer(
        dataframe=df,
        weight_column="weights",
        energy_column="energy_mev",
    )
    phase_space.savefig()

    # Apply thinning algorithm to df, resulting in df_thin
    logger.info("Reducing number of particles.\n")
    resampler = ParticleResampler(
        dataframe=df,
        weight_column="weights",
    )
    df_thin = resampler.global_leveling_thinning().set_weights_to(1).finalize()
    dataset_info(df_thin)

    # Visualize both dataframes in order to see effects of thining
    phase_space_thin = PhaseSpaceVisualizer(df_thin)
    phase_space.save_comparative_fig(phase_space_thin)

    # Write the reduced dataframe to a file
    DataFrameToFile(df_thin).exclude_weights().exclude_energy().write_to_file(
        opmd_path.with_suffix(".txt")
    )


if __name__ == "__main__":
    main()
