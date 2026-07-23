"""
Module: df_to_txt
This module provides a class for writing pandas DataFrame to a text file with custom headers.
"""

import os

import pandas as pd

from .log import logger
from .units import constants, units
from .utils import format_file_size

MOMENTUM_COLUMNS = ["momentum_x_mev_c", "momentum_y_mev_c", "momentum_z_mev_c"]


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
        self.momentum_divisor = None
        self.momentum_unit = "m*c"

    def exclude_weights(self):
        self.include_weights = False
        return self

    def exclude_energy(self):
        self.include_energy = False
        return self

    def momentum_in_mc(self, mass_mev_c2: float):
        """
        Write momenta as normalized momentum u = p / (m c) instead of MeV/c,
        where m is the particle species mass (openpmd_viewer's 'ux' convention).
        For massless species (mass_mev_c2 = 0), u = p / (m c) is undefined, so
        the momenta are normalized by the electron mass instead, u = p / (m_e c),
        the usual PIC convention for photon momenta (Smilei, PIConGPU).
        """
        if mass_mev_c2 < 0:
            raise ValueError("momentum_in_mc requires a non-negative species mass.")
        if mass_mev_c2 == 0:
            self.momentum_divisor = constants.electron_mass_mev_c2
            self.momentum_unit = "m_e*c"
        else:
            self.momentum_divisor = mass_mev_c2
        return self

    def write_to_file(self, file_path, fortran_unformatted=False):
        columns_to_write = self.df.columns.tolist()
        if not self.include_weights:
            columns_to_write.remove("weights")
        if not self.include_energy:
            columns_to_write.remove("kinetic_energy_mev")

        logger.info("Writing dataframe to file. This may take a while...\n")
        if fortran_unformatted:
            self._write_fortran_unformatted(file_path, columns_to_write)
        else:
            self._write_csv(file_path, columns_to_write)
        logger.info("Wrote %s\n", file_path)

        file_size = os.path.getsize(file_path)
        logger.info("Final file size: %s\n", format_file_size(file_size))

    def _column_data(self, column):
        data = self.df[column]
        if self.momentum_divisor is not None and column in MOMENTUM_COLUMNS:
            # float32 / python float stays float32
            data = data / self.momentum_divisor
        return data

    def _column_unit(self, column):
        if self.momentum_divisor is not None and column in MOMENTUM_COLUMNS:
            return self.momentum_unit
        return self.units[column]

    def _write_csv(self, file_path, columns_to_write, chunk_size=5_000_000):
        with open(file_path, "w", encoding="utf-8") as f:
            header = ", ".join(
                f"{column} ({self._column_unit(column)})" for column in columns_to_write
            )
            f.write(header + "\n")

            for start in range(0, len(self.df), chunk_size):
                chunk = self.df.iloc[start : start + chunk_size][columns_to_write].copy()
                if self.momentum_divisor is not None:
                    for column in MOMENTUM_COLUMNS:
                        if column in chunk.columns:
                            chunk[column] = chunk[column] / self.momentum_divisor
                chunk.to_csv(
                    f,
                    index=False,
                    header=False,
                    sep=",",
                    float_format="%.7e",
                )

    def _write_fortran_unformatted(self, file_path, columns_to_write):
        import numpy as np
        from scipy.io import FortranFile

        # The Fortran sequential format stores n as int32 and each record's
        # byte count as a uint32 marker (scipy wraps silently instead of
        # raising), so one float32 column record caps the particle count.
        max_rows = (2**32 - 1) // 4
        if len(self.df) > max_rows:
            raise ValueError(
                f"{len(self.df):,} particles exceed the Fortran unformatted"
                f" limit of {max_rows:,} (one 4 GiB record per float32 column);"
                " reduce the particle count or write CSV instead."
            )

        with FortranFile(file_path, "w") as f:
            f.write_record(np.array([len(self.df)], dtype=np.int32))
            for col in columns_to_write:
                f.write_record(self._column_data(col).to_numpy().astype(np.float32))
