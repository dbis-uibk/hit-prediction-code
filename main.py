#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold


def setup_argument_parser():
    """Configures the argument pareser."""

    parser = argparse.ArgumentParser(
        description='Exploring hit song predictions.')

    parser.add_argument(
        '-d, --dataset',
        type=str,
        help='name of the dataset',
        dest='dataset',
        required=True
    )

    parser.add_argument(
        '-c, --config-file',
        type=str,
        help='path to config file',
        dest='config_file',
        default="./config/config.json"
    )

    return parser.parse_args()


def main():
    print("Hallo Hit Songs!")

if __name__ == '__main__':
    main()
