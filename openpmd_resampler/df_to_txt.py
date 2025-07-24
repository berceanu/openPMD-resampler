"""
Module: df_to_txt
This module provides a class for writing pandas DataFrame to a text file with custom headers.
"""

import os

import pandas as pd

from .log import logger
from .units import units
from .utils import convert_bytes_to_mb, convert_bytes_to_gb


class DataFrameToFile:
    """
    A class used to write a pandas DataFrame to a text file with a custom header.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The pandas DataFrame to be written to a text file.
        """
        self.df = df
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
        with open(file_path, "w", encoding="utf-8") as f:
            header = ", ".join(
                f"{column} ({self.units[column]})" for column in columns_to_write
            )
            f.write(header + "\n")

        # Append DataFrame to file
        logger.info("Writing dataframe to file. This may take a while...\n")
        self.df.to_csv(
            file_path,
            columns=columns_to_write,
            index=False,
            header=False,
            sep=",",
            float_format="%.7e",
            mode="a",
        )
        logger.info("Wrote %s\n", file_path)

        # Compute and log the file size
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = convert_bytes_to_mb(file_size_bytes)
        logger.info("Final file size: %.2f MB", file_size_mb)
