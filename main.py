#!/usr/bin/env python3

import git
import json
from datetime import datetime
import pandas as pd
from dbispipeline.core import Core as Pipeline
from dbispipeline.core import load_config, setup_argument_parser


def main():
    print("Hallo Hit Songs!")

    repo = git.Repo(search_parent_directories=True)
    sha_commit_id = repo.head.object.hexsha
    print(sha_commit_id)
    print(repo.remotes.origin.url)
    print(repo.is_dirty())

    pipeline = Pipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
