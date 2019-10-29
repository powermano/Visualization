import argparse
from importlib import import_module


def parse_args():
    """An easy method get config file.
    """
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument(
        '--cfg', help='experiment configure file path', required=True, type=str)

    return parser.parse_args()
