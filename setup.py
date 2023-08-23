from setuptools import setup, find_packages

setup(
    name="opmdtogeant",
    version="0.1.0",
    description="A Python package to export particle data from openPMD to text format",
    author="Andrei Berceanu",
    author_email="andreicberceanu@gmail.com",
    packages=find_packages(),
    python_requires="==3.11.*",
    install_requires=[
        "openpmd-api==0.15.*",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)

# TODO: update requirements
