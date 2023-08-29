"""
This module provides various utility functions.
# TODO: split into math_utils, img_utils etc
"""

import tempfile
from typing import List

from PIL import Image


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

    final_image.save(output_filename)
