"""This module contains common functions and settings."""
import os.path
import re

import numpy as np
import pandas as pd

cache = {'model': None, 'X': None, 'y': None}


def peak_pos_non_hit_value():
    """Returns: the value used for non-hits when predicting peak pos."""
    return 101


def peak_position_labels():
    """Returns: the labels used to map regression values."""
    return list(range(1, 102))


def weeks_non_hit_value():
    """Returns: the value used for non-hits when predicting weeks."""
    return 0


def weeks_labels():
    """
    The range is chosen based on the fact, that billboard removes songs form
    hot 100 after 20 weeks if the song is below position 50.
    See:
    Music Popularity: Metrics, Characteristics, and Audio-Based Prediction
    Junghyuk Lee and Jong-Seok Lee , Senior Member, IEEE

    Returns: the labels used to map regression values.
    """
    return list(range(0, 21))


def _exp_labels(exp_min, exp_max, steps, base=2):
    return list(
        np.logspace(
            exp_min,
            exp_max,
            num=steps,
            base=base,
        ).astype(np.int))


def _linear_labels(exp_min, exp_max, steps, base=2):
    min_value = base**exp_min
    max_value = base**exp_max
    step_size = int((max_value - min_value) / (steps + 1))

    return list(range(min_value, max_value, step_size))


def _lfmlc_exps_and_steps():
    return (6, 20, 101, 2)


def lfmlc_labels():
    """
    The range is chosen to be base two logarithmic.

    Maximum listener count is approximately 2M.
    """
    return _exp_labels(*_lfmlc_exps_and_steps())


def linear_lfmlc_labels():
    """
    The range is chosen to be linear.

    Maximum listener count is approximately 2M.
    """
    return _linear_labels(*_lfmlc_exps_and_steps())


def _lfmpc_exps_and_steps():
    return (8, 23, 101, 2)


def lfmpc_labels():
    """
    The range is chosen to be base two logarithmic.

    Maximum play count is approximately 20M.
    """
    return _exp_labels(*_lfmpc_exps_and_steps())


def linear_lfmpc_labels():
    """
    The range is chosen to be linear.

    Maximum play count is approximately 20M.
    """
    return _linear_labels(*_lfmpc_exps_and_steps())


def wide_and_deep_epochs():
    """Returns: the number of epochs used to train wide and deep."""
    return 2500


def read_dataframe(path, **kwargs):
    """Reads a pandas dataframe from a file.

    It supports multiple file extensions.

    Args:
        path: the path to the file.
        kwargs: key word arguments passed to the respective read function.

    Retruns: the dataframe.
    """
    filename, file_ext = os.path.splitext(path)

    if file_ext == '.xz':
        _, file_ext2 = os.path.splitext(filename)
        file_ext = file_ext2 + file_ext

    if file_ext == '.pickle':
        return pd.read_pickle(path, **kwargs)
    if file_ext == '.pickle.xz':
        return pd.read_pickle(path, 'xz', **kwargs)
    elif file_ext == '.parquet':
        return pd.read_parquet(path, **kwargs)
    elif file_ext == '.csv':
        return pd.read_csv(path, header=0, index_col=0, **kwargs)
    else:
        raise ValueError('File extension %s unknown.' % file_ext)


def cached_model_predict(model, X):
    if cache['model'] != model or not np.array_equal(cache['X'], X):
        cache['model'] = model
        cache['X'] = X
        cache['y'] = model.predict(X)
    return cache['y']


def cached_model_predict_clear():
    cache['model'] = None
    cache['X'] = None
    cache['y'] = None


def feature_columns(columns, feature_select):
    try:
        feature, part = feature_select
    except ValueError:
        feature, part = (feature_select, None)

    if feature == 'hl':
        regex_filter = highlevel_regex()
    elif feature == 'genre':
        regex_filter = genre_regex()
    elif feature == 'mood':
        regex_filter = mood_regex()
    elif feature == 'voice':
        regex_filter = voice_regex()
    elif feature == 'ismir04_rhythm':
        regex_filter = ismir04_rhythm_regex()
    elif feature == 'moods_mirex':
        regex_filter = moods_mirex_regex()
    elif feature == 'hl_rhythm':
        regex_filter = hl_rhythm_regex()
    elif feature == 'hl_tonal':
        regex_filter = hl_tonal_regex()
    elif feature == 'll_beats_loudness':
        regex_filter = ll_beats_loudness_regex()
    elif feature == 'll_bpm_histogram':
        regex_filter = ll_bpm_histogram_regex()
    else:
        regex_filter = feature

    return filter_features(columns, regex_filter), part


def filter_features(columns, regex_filter):
    regex = re.compile(regex_filter)
    return list(filter(regex.match, columns))


def highlevel_regex():
    return r'highlevel\.\w+\.all\.\w+'


def lowlevel_regex():
    return r'lowlevel\.\w+\.\w+'


def genre_regex():
    return r'highlevel\.genre_tzanetakis\.all\.\w+'


def mood_regex():
    return r'highlevel\.mood_\w+\.all\.[a-z]+$'


def voice_regex():
    return r'(highlevel\.voice_instrumental\.all\.\w+|highlevel\.gender\.all\.\w+)'


def ismir04_rhythm_regex():
    return r'highlevel\.ismir04_rhythm\.all\.\w+'


def moods_mirex_regex():
    return r'highlevel\.moods_mirex\.all\.\w+'


def danceability_regex():
    return r'highlevel\.danceability\.all\.danceable'


def tonal_atonal_regex():
    return r'highlevel\.tonal_atonal\.all\.\w+'


def hl_rhythm_regex():
    return r'rhythm\.(bpm|beats_count|onset_rate|danceability)$'


def hl_tonal_regex():
    return r'tonal\.(chords_changes_rate|chords_number_rate|tuning_frequency|key_strength|tuning_diatonic_strength|tuning_equal_tempered_deviation|tuning_nontempered_energy_ratio)'


def ll_tonal_regex():
    return r'tonal\.(chords_strength|hpcp_entropy)\.\w+'


def ll_beats_loudness_regex():
    return r'rhythm\.beats_loudness\.\w+$'


def ll_bpm_histogram_regex():
    # FIXME: selechts multiple values but only some are meaningfull
    return r'rhythm\.bpm_histogram_\w+\.\w+'


def mood_list():
    return [
        ('mood', 'wide'),
        ('moods_mirex', 'wide'),
    ]


def genre_list():
    return [
        ('genre', 'wide'),
    ]


def voice_list():
    return [
        ('voice', 'wide'),
    ]


def hl_list():
    return [
        *hl_no_year_list(),
        ('year', 'wide'),
    ]


def hl_no_year_list():
    return [
        *mood_list(),
        *genre_list(),
        *voice_list(),
    ]


def all_ll_genre_list():
    return [
        *ll_list(),
        *genre_list(),
    ]


def all_ll_mood_list():
    return [
        *ll_list(),
        *mood_list(),
    ]


def all_ll_voice_list():
    return [
        *ll_list(),
        *voice_list(),
    ]


def all_ll_year_list():
    return [
        *ll_list(),
        ('year', 'wide'),
    ]


def all_ll_year_genre_list():
    return [
        *all_ll_genre_list(),
        ('year', 'wide'),
    ]


def all_ll_year_mood_list():
    return [
        *all_ll_mood_list(),
        ('year', 'wide'),
    ]


def all_ll_year_voice_list():
    return [
        *all_ll_voice_list(),
        ('year', 'wide'),
    ]


def all_ll_year_genre_mood_list():
    return [
        *all_ll_year_genre_list(),
        *mood_list(),
    ]


def all_ll_year_genre_voice_list():
    return [
        *all_ll_year_genre_list(),
        *voice_list(),
    ]


def all_ll_year_mood_voice_list():
    return [
        *all_ll_year_mood_list(),
        *voice_list(),
    ]


def rhythm_list():
    return [
        (hl_rhythm_regex(), 'deep'),
        (ll_beats_loudness_regex(), 'deep'),
        (ll_bpm_histogram_regex(), 'deep'),
    ]


def chords_list():
    return [
        (hl_tonal_regex(), 'deep'),
        (ll_tonal_regex(), 'deep'),
        (r'tonal\.chords_(key|scale).*', 'deep'),
        (r'tonal\.key_(key|scale).*', 'deep'),
    ]


def yang_list():
    return [
        (r'lowlevel\.dissonance\.\w+', 'deep'),
        (r'lowlevel\.spectral_centroid\.\w+', 'deep'),
        (r'lowlevel\.average_loudness', 'deep'),
        (r'lowlevel\.spectral_rolloff\.\w+', 'deep'),
        (r'lowlevel\.spectral_kurtosis\.\w+', 'deep'),
        (r'lowlevel\.barkbands_skewness\.\w+', 'deep'),
        (r'lowlevel\.barkbands_spread\.\w+', 'deep'),
        (r'lowlevel\.spectral_flux\.\w+', 'deep'),
        (r'lowlevel\.barkbands_flatness_db\.\w+', 'deep'),
        (r'lowlevel\.spectral_energyband_low\.\w+', 'deep'),
    ]


def lowlevel_list():
    return [
        (r'lowlevel\.average_loudness', 'deep'),
        (r'lowlevel\.dissonance\.\w+', 'deep'),
        (r'lowlevel\.barkbands_(crest|kurtosis|skewness|spread)\.\w+', 'deep'),
        (r'lowlevel\.barkbands_flatness_db\.\w+', 'deep'),
        (r'lowlevel\.dynamic_complexity', 'deep'),
        (r'lowlevel\.erbbands_(crest|kurtosis|skewness|spread)\.\w+', 'deep'),
        (r'lowlevel\.erbbands_flatness_db\.\w+', 'deep'),
        (r'lowlevel\.hfc\.\w+', 'deep'),
        (r'lowlevel\.melbands_flatness_db\.\w+', 'deep'),
        (r'lowlevel\.melbands_(crest|kurtosis|skewness|spread)\.\w+', 'deep'),
        (r'lowlevel\.pitch_salience\.\w+', 'deep'),
        (r'lowlevel\.silence_rate_(2|3|6)0dB\.\w+', 'deep'),
        (r'lowlevel\.spectral_centroid\.\w+', 'deep'),
        (r'lowlevel\.spectral_complexity\.\w+', 'deep'),
        (r'lowlevel\.spectral_decrease\.\w+', 'deep'),
        (r'lowlevel\.spectral_energy\.\w+', 'deep'),
        (r'lowlevel\.spectral_energyband_(high|middle_high|middle_low|low)\.\w+',
         'deep'),
        (r'lowlevel\.spectral_entropy\.\w+', 'deep'),
        (r'lowlevel\.spectral_flux\.\w+', 'deep'),
        (r'lowlevel\.spectral_(kurtosis|skewness|spread)\.\w+', 'deep'),
        (r'lowlevel\.spectral_rms\.\w+', 'deep'),
        (r'lowlevel\.spectral_rolloff\.\w+', 'deep'),
        (r'lowlevel\.spectral_strongpeak\.\w+', 'deep'),
        (r'lowlevel\.zerocrossingrate\.\w+', 'deep'),
    ]


def ll_filterd_list():
    return [
        *yang_list(),
        *chords_list(),
        *rhythm_list(),
    ]


def ll_list():
    return [
        *lowlevel_list(),
        *chords_list(),
        *rhythm_list(),
    ]


def all_list():
    return [
        *hl_list(),
        *ll_list(),
    ]


def all_no_year_list():
    return [
        *hl_no_year_list(),
        *ll_list(),
    ]


def all_filtered_list():
    return [
        *hl_list(),
        *ll_filterd_list(),
    ]


def get_columns_matching_list(columns, filter_list):
    matching_cols = []
    for filter_entry in filter_list:
        cols, _ = feature_columns(columns, filter_entry)
        matching_cols += cols

    return set(matching_cols)
