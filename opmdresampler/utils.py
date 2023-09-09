"""
This module provides various utility functions.
"""
import os
import tempfile
from typing import List

import pandas as pd
from PIL import Image

from opmdresampler.log import logger
from opmdresampler.units import constants


def convert_bytes_to_gb(size_in_bytes):
    """Convert size in bytes to gigabytes (GB)"""
    return size_in_bytes / (1024.0**3)


def thousand_separators(number):
    return f"{number:,}"


def dataset_info(df: pd.DataFrame) -> None:
    logger.info(
        "The dataset contains %s macroparticles, corresponding to %s 'real' electrons, "
        "with a total charge of %.2f pC.\n",
        thousand_separators(df.shape[0]),
        thousand_separators(int(df["weights"].sum())),
        int(df["weights"].sum()) * constants.electron_charge_picocoulombs,
    )
    logger.info("Descriptive statistics of the dataset:\n")
    logger.info("```\n")
    logger.info("%s\n", df.describe())
    logger.info("```\n")


def unique_filename(suffix: str) -> str:
    """
    This function generates a unique filename with a specified suffix.
    The file is created immediately upon calling this function, and is not automatically deleted
    when the function exits. Remember to delete it later via ``os.remove``.

    Parameters:
    suffix (str): The suffix to append to the filename.

    Returns:
    str: The unique filename generated.
    """

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_filename = temp_file.name
    return temp_filename


def combine_images(filenames: List[str], output_filename: str) -> None:
    """
    This function combines multiple images into a single image vertically.
    All images are assumed to have the same width.
    The images are combined in the order they are provided in the list of filenames.

    Parameters:
    filenames (List[str]): List of filenames of the images to be combined.
    output_filename (str): The filename of the output image.

    Returns:
    None
    """

    images = [Image.open(filename) for filename in filenames]

    # Assume all images have the same width
    width = images[0].width
    # Height of final image
    height = sum(image.height for image in images)

    final_image = Image.new("RGB", (width, height))

    # Paste the images onto the final image vertically
    y_offset = 0
    for image in images:
        final_image.paste(image, (0, y_offset))
        y_offset += image.height

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    final_image.save(output_filename)
    logger.info("Wrote %s\n", output_filename)

    def log_link(filename):
        logger.info(f'<a href="{filename}"><img src="{filename}" width="200"></a>\n')

    log_link(output_filename)
