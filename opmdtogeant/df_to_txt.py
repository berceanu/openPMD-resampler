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
        # Mapping for special cases
        special_cases = {
            "um": "[um]",
            "mev_c": "[MeV/c]",
            "mev": "[MeV]",
        }

        header = []
        for name in self.dataframe.columns:
            parts = name.split("_")
            # Convert special cases
            for i, part in enumerate(parts):
                if part in special_cases:
                    parts[i] = special_cases[part]
            # Join the parts back together and add to header
            header.append(" ".join(parts))

        # Combine all headers into a single string
        header_str = "# " + " ".join(header)
        return header_str

    def write_to_file(self, filename: str) -> None:
        """
        Writes the DataFrame to a text file with a custom header.

        Parameters
        ----------
        filename : str
            The name of the text file to be written.
        """
        header = self._create_header()

        # Write DataFrame to file without header
        self.dataframe.to_csv(filename, sep=" ", index=False, header=False)

        # Add custom header to file
        with open(filename, "r+") as file:
            content = file.read()
            file.seek(0, 0)
            file.write(header.rstrip("\r\n") + "\n" + content)
