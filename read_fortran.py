"""
Read the Fortran unformatted (sequential) particle file written by
`start.py --fortran_unformatted` / `usage.py --fortran_unformatted`.

File layout (each record framed by uint32 byte-count markers):
    record 0 : int32[1]    number of particles N
    record 1+: float32[N]  one record per column, in DataFrame column order

With the default writer settings (start.py: energy excluded, weights kept,
momenta normalized) the columns are
    x, y, z [m], ux, uy, uz [p/(m c)], w [macroparticle weight].
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import FortranEOFError, FortranFile

# Column names by number of float32 records, matching DataFrameToFile.
COLUMNS_BY_COUNT = {
    6: ["x_m", "y_m", "z_m", "ux", "uy", "uz"],
    7: ["x_m", "y_m", "z_m", "ux", "uy", "uz", "weights"],
    8: ["x_m", "y_m", "z_m", "ux", "uy", "uz", "weights", "kinetic_energy_mev"],
}


def read_fortran_particles(file_path) -> pd.DataFrame:
    """Return the particle columns of a Fortran unformatted file as a DataFrame."""
    records = []
    with FortranFile(file_path, "r") as f:
        n = int(f.read_ints(np.int32)[0])
        while True:
            try:
                records.append(f.read_reals(np.float32))
            except FortranEOFError:
                break

    for i, record in enumerate(records):
        if record.size != n:
            raise ValueError(f"record {i + 1} has {record.size} values, expected {n}")

    names = COLUMNS_BY_COUNT.get(len(records), [f"col{i}" for i in range(len(records))])
    return pd.DataFrame(dict(zip(names, records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("file", type=Path, help="Fortran unformatted .dat file")
    parser.add_argument("--csv", type=Path, help="optionally save the data as CSV")
    args = parser.parse_args()

    df = read_fortran_particles(args.file)
    print(f"{args.file}: {len(df):,} particles, {df.shape[1]} columns")
    print(df.describe().T.to_string())
    if "weights" in df:
        print(f"\nTotal weight (physical particles): {df['weights'].astype(np.float64).sum():.6e}")

    if args.csv:
        df.to_csv(args.csv, index=False, float_format="%.7e")
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
