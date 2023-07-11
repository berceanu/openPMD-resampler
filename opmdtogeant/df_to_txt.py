"""
Module: df_to_txt
This module provides a class for writing pandas DataFrame to a text file with custom headers.
"""

import pandas as pd


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
        self.dataframe = dataframe

    def _create_header(self) -> str:
        """
        Private method to create a custom header string from DataFrame column names.

        Returns
        -------
        str
            A string that represents the custom header for the text file.
        """
        # Hard-coded mapping from dataframe's column names to the desired header labels
        column_name_mapping = {
            "position_x_um": "position x [um]",
            "position_y_um": "position y [um]",
            "position_z_um": "position z [um]",
            "momentum_x_mev_c": "momentum x [MeV/c]",
            "momentum_y_mev_c": "momentum y [MeV/c]",
            "momentum_z_mev_c": "momentum z [MeV/c]",
            "weights": "weights",
            "energy_mev": "energy [MeV]",
        }

        # Create header
        header = [column_name_mapping[col] for col in self.dataframe.columns]

        # Combine all headers into a single string
        header_str = "# " + ",".join(header)
        return header_str

    def write_to_file(self, filename: str) -> None:
        """
        Writes the DataFrame to a text file with a custom header.

        Parameters
        ----------
        filename : str
            The name of the text file to be written.
        """
        print("Writing dataframe to file. This may take a while...")
        header = self._create_header()

        # Write DataFrame to file without header
        self.dataframe.to_csv(
            filename, sep=",", index=False, header=False, float_format="%.16e"
        )

        # Add custom header to file
        with open(filename, "r+") as file:
            content = file.read()
            file.seek(0, 0)
            file.write(header.rstrip("\r\n") + "\n" + content)
