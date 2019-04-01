#!/usr/bin/env python3

from dataloaders import MsdBbLoader


def main():
    loader = MsdBbLoader(
        hits_file_path=
        '/storage/nas3/datasets/music/billboard/msd_bb_matches.csv',
        non_hits_file_path=
        '/storage/nas3/datasets/music/billboard/msd_bb_non_matches.csv',
        features_path=
        '/storage/nas3/datasets/music/billboard',
        non_hits_per_hit=1,
    )

    print(loader.load().shape)
    print(loader.configuration())


if __name__ == '__main__':
    main()
