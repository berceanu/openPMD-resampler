import argparse
from pathlib import Path

from opmdtogeant.df_to_txt import DataFrameToFile
from opmdtogeant.reader import HDF5Reader, electron_mass_MeV_c2
from opmdtogeant.visualize_phase_space import PhaseSpaceVisualizer
from opmdtogeant.resampling import ParticleResampler


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()
    h5_path = Path(args.h5_path)

    # Create the dataframe
    h5_reader = HDF5Reader(h5_path, "e_highGamma")  # e_highGamma or e_all
    df = h5_reader.build_df()

    # Write the dataframe to a file
    # df_to_file = DataFrameToFile(df)
    # TODO: uncomment
    # df_to_file.write_to_file(h5_path.with_suffix(".txt")) # slow

    # Create the phase space plots
    phase_space = PhaseSpaceVisualizer(
        dataframe=df,
        weight_column="weights",
        energy_column="energy_mev",
    )
    phase_space.savefig()

    # apply thinning algorithm to df, resulting in df_thin
    resampler = ParticleResampler(
        dataframe=df,
        weight_column="weights",
    )
    # df_thin = resampler.set_weights_to(1)
    df_thin = resampler.simple_thinning(number_of_remaining_particles=10**5)
    print(df_thin.describe())
    print(
        f"The dataset contains {df_thin.shape[0]:,} macroparticles, corresponding to {int(df_thin['weights'].sum()):,} 'real' electrons"
    )

    phase_space_thin = PhaseSpaceVisualizer(
        dataframe=df_thin,
        weight_column="weights",
        energy_column="energy_mev",
    )

    # visualize both dataframes in order to see effects of thining
    phase_space.save_comparative_fig(phase_space_thin)


if __name__ == "__main__":
    main()
