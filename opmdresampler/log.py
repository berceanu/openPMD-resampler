"""
This module provides a function to set up a logger with a specified name, log file, and level.
The logger can be used to log messages to both the console and a file.
"""
import logging
import os
from typing import Union


def setup_logger(
    name: str, log_file: Union[str, os.PathLike], level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with specified name, log file and level.

    Args:
        name (str): The name of the logger.
        log_file (Union[str, os.PathLike]): The file to which the logger will write.
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger.
    """
    # Create a custom logger
    custom_logger = logging.getLogger(name)

    # Set the level for the logger
    custom_logger.setLevel(level)

    # Remove all handlers if any exist
    custom_logger.handlers = []

    # Remove the log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create formatters and add it to handlers
    c_format = logging.Formatter("%(message)s")
    f_format = logging.Formatter("%(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    custom_logger.addHandler(c_handler)
    custom_logger.addHandler(f_handler)

    return custom_logger


# Use the setup_logger function to set up a logger
logger = setup_logger("opmdresampler", os.path.join(os.getcwd(), "output.md"))
