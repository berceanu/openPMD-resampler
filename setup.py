from setuptools import setup, find_packages

setup(
    name="opmdtogeant",
    version="0.1.0",
    description="A Python package to read, parse, merge, and write data from HDF5 files produced by PIConGPU",
    author="Andrei Berceanu",
    author_email="andreicberceanu@gmail.com",
    packages=find_packages(),
    install_requires=[
        "openpmd-api==0.14.2",
        "h5py==3.3.0",
        "numpy==1.21.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)

