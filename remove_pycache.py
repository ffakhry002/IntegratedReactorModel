#!/usr/bin/env python3
"""Script to remove all __pycache__ directories recursively."""

import os
import shutil
import sys

def remove_pycache():
    """Remove all __pycache__ directories in the current directory and subdirectories.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Get the starting directory (where the script is run from)
    start_dir = os.getcwd()

    print(f"Searching for __pycache__ directories in: {start_dir}")
    removed_count = 0

    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed: {pycache_path}")
                removed_count += 1
            except Exception as e:
                print(f"Error removing {pycache_path}: {e}", file=sys.stderr)

    if removed_count > 0:
        print(f"\nSuccessfully removed {removed_count} __pycache__ directories")
    else:
        print("\nNo __pycache__ directories found")

if __name__ == '__main__':
    remove_pycache()
