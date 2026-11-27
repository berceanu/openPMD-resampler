"""
Read the Fortran unformatted (sequential) particle file written by
`start.py --fortran_unformatted` / `usage.py --fortran_unformatted`.

File layout (each record framed by uint32 byte-count markers):
    record 0 : int32[1]    number of particles N
    record 1+: float32[N]  one record per column, in DataFrame column order

With the default writer settings (start.py: energy excluded, weights kept,
momenta normalized) the columns are
    x, y, z [m], ux, uy, uz [p/(m c)], w [macroparticle weight].

The file does not store column names, so without --columns they are guessed
from the number of records (COLUMNS_BY_COUNT). Seven records are ambiguous: the
default layout above, or a file written with exclude_weights() that kept the
kinetic energy. Pass the names for such files, e.g.
    --columns x_m y_m z_m ux uy uz kinetic_energy_mev
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.io import FortranEOFError, FortranFile

# Column names by number of float32 records, matching DataFrameToFile with
# start.py's layout (weights before energy, energy the first to be dropped).
COLUMNS_BY_COUNT = {
    6: ["x_m", "y_m", "z_m", "ux", "uy", "uz"],
    7: ["x_m", "y_m", "z_m", "ux", "uy", "uz", "weights"],
    8: ["x_m", "y_m", "z_m", "ux", "uy", "uz", "weights", "kinetic_energy_mev"],
}


def read_fortran_particles(file_path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Return the particle columns of a Fortran unformatted file as a DataFrame.

    columns: names of the float32 records in file order. By default they are
        guessed from the record count (COLUMNS_BY_COUNT, start.py's layout);
        give them explicitly for files written with a different layout, such
        as exclude_weights() with the energy kept, which also has 7 records.
    """
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

    if columns is None:
        names = COLUMNS_BY_COUNT.get(len(records), [f"col{i}" for i in range(len(records))])
    else:
        names = list(columns)
        if len(names) != len(records):
            raise ValueError(
                f"{file_path} has {len(records)} data records, but {len(names)}"
                f" column names were given: {names}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"column names must be unique, got {names}")
    return pd.DataFrame(dict(zip(names, records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("file", type=Path, help="Fortran unformatted .dat file")
    parser.add_argument(
        "--columns", nargs="+", metavar="NAME",
        help="names of the data records in file order (default: guessed from the record"
             " count, assuming start.py's layout); needed for a 7-record file written"
             " without weights but with energy: x_m y_m z_m ux uy uz kinetic_energy_mev",
    )
    parser.add_argument("--csv", type=Path, help="optionally save the data as CSV")
    args = parser.parse_args()

    df = read_fortran_particles(args.file, columns=args.columns)
    print(f"{args.file}: {len(df):,} particles, {df.shape[1]} columns")
    print(df.describe().T.to_string())
    if "weights" in df:
        print(f"\nTotal weight (physical particles): {df['weights'].astype(np.float64).sum():.6e}")

    if args.csv:
        df.to_csv(args.csv, index=False, float_format="%.7e")
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
