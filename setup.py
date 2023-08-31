"""
This is a setup file for the opmdtogeant package.
"""

from setuptools import setup, find_packages


with open("requirements.txt", encoding="utf-8") as f:
    required = f.read().splitlines()

setup(
    name="opmdtogeant",
    version="0.1.0",
    description="A Python package to export particle data from openPMD to text format",
    author="Andrei Berceanu",
    author_email="andreicberceanu@gmail.com",
    packages=find_packages(),
    python_requires="==3.11.*",
    install_requires=required,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
