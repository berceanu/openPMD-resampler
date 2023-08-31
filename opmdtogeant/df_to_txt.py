"""
Module: df_to_txt
This module provides a class for writing pandas DataFrame to a text file with custom headers.
"""

import pandas as pd
from opmdtogeant.units import units


class DataFrameToFile:
    """
    A class used to write a pandas DataFrame to a text file with a custom header.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            The pandas DataFrame to be written to a text file.
        """
        self.df = dataframe
        self.units = units
        self.include_weights = True
        self.include_energy = True

    def exclude_weights(self):
        self.include_weights = False
        return self

    def exclude_energy(self):
        self.include_energy = False
        return self

    def write_to_file(self, file_path):
        # Select columns to write based on include_weights and include_energy
        columns_to_write = self.df.columns.tolist()
        if not self.include_weights:
            columns_to_write.remove("weights")
        if not self.include_energy:
            columns_to_write.remove("energy_mev")

        # Write header to file
        with open(file_path, "w") as f:
            for column in columns_to_write:
                f.write(f"{column} ({self.units[column]}), ")
            f.write("\n")

        # Append DataFrame to file
        print("Writing dataframe to file. This may take a while...", end=" ")
        self.df.to_csv(
            file_path,
            columns=columns_to_write,
            index=False,
            header=False,
            sep=",",
            float_format="%.16e",
            mode="a",
        )
        print(f"Wrote {file_path}.")
