#!/usr/bin/env python3

from .dataloaders import MsdBbLoader


def main():
    loader = MsdBbLoader(
        hits_file_path=
        'data/hit_song_prediction_ismir2020/processed/msd_bb_matches.csv',
        non_hits_file_path=
        'data/hit_song_prediction_ismir2020/processed/msd_bb_non_matches.csv',
        features_path='data/processed',
        non_hits_per_hit=1,
        features=['hl'],
        label='weeks',
    )

    print(loader.load()[0][0].shape, loader.load()[1].shape)
    print(loader.configuration())
    print(loader.load()[0][0].columns)


if __name__ == '__main__':
    main()
