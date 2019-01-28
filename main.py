#!/usr/bin/env python3

import git
import json
from datetime import datetime
import pandas as pd
from dbispipeline.core import Core as Pipeline
from dbispipeline.core import load_config, setup_argument_parser


def main():
    print("Hallo Hit Songs!")

    repo = git.Repo('./')
    print(repo.is_dirty())

    pipeline = Pipeline()
    pipeline.run()

if __name__ == '__main__':
    main()
