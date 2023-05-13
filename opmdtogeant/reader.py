import h5py
from openpmd_api import Series, Access_Type, Iteration_Encoding

class HDF5Reader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        try:
            series = Series(self.file_path, Access_Type.read_only)

            # Get the last iteration
            it = series.iterations[sorted(series.iterations.keys())[-1]]

            # Get the electrons particle species
            electrons = it.particles['electrons']

            # Position and momenta data
            position_x = electrons['position']['x'][:]
            position_y = electrons['position']['y'][:]
            position_z = electrons['position']['z'][:]

            momenta_px = electrons['momentum']['x'][:]
            momenta_py = electrons['momentum']['y'][:]
            momenta_pz = electrons['momentum']['z'][:]

            # Weight data
            weights = electrons['weighting'][:]

            return position_x, position_y, position_z, momenta_px, momenta_py, momenta_pz, weights

        except Exception as e:
            print(f"An error occurred while reading the file: {e}")

