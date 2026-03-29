# -*- coding: utf-8 -*-
"""
Utility functions for the project.
"""

import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def create_directories(dirs_dict):
    """Creates all directories specified in the config dictionary."""
    for path in dirs_dict.values():
        try:
            os.makedirs(path, exist_ok=True)
            logging.info(f"Directory verified/created: {path}")
        except OSError as e:
            logging.error(f"Error creating directory {path}: {e}")
            raise

def get_logger(name):
    """Returns a logger instance."""
    return logging.getLogger(name)
