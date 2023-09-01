"""
This module provides a command-line interface for reading HDF5 files, 
visualizing phase space, resampling particles, and writing the results to a text file.
"""
import argparse
import sys
from pathlib import Path

from opmdtogeant.df_to_txt import DataFrameToFile
from opmdtogeant.log import logger
from opmdtogeant.reader import HDF5Reader
from opmdtogeant.resampling import ParticleResampler
from opmdtogeant.utils import dataset_info
from opmdtogeant.visualize_phase_space import PhaseSpaceVisualizer


def main():
    """
    Main function to parse command line arguments, read HDF5 files, visualize phase space,
    resample particles, and write the results to a text file.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()
    h5_path = Path(args.h5_path)

    # Log the command used to run the script
    command = " ".join(sys.argv)
    logger.info("## Output\n")
    logger.info("This is the output of `%s`\n", command)

    # Create the dataframe
    h5_reader = HDF5Reader(h5_path, "e_highGamma")  # "e_highGamma" or "e_all"
    df = h5_reader.build_df()

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
        h5_path.with_suffix(".txt")
    )


if __name__ == "__main__":
    main()
