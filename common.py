import re


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
    return r'tonal\.(chords_changes_rate|chords_number_rate|tuning_frequency)'


def ll_beats_loudness_regex():
    return r'rhythm.beats_loudness\.\w+$'


def ll_bpm_histogram_regex():
    # FIXME: selechts multiple values but only some are meaningfull
    return r'rhythm\.bpm_histogram_\w+\.\w+'


def yang_list():
    return [
        (r'lowlevel\.dissonance\.\w+', 'deep'),
        (r'lowlevel\.spectral_centroid\.\w+', 'deep'),
        (r'lowlevel\.average_loudness', 'wide'),
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
        (r'lowlevel\.average_loudness', 'wide'),
        (r'lowlevel\.dissonance\.\w+', 'deep'),
        (r'lowlevel\.barkbands_(crest|kurtosis|skewness|spread)\.\w+', 'deep'),
        (r'lowlevel\.barkbands_flatness_db\.\w+', 'deep'),
        (r'lowlevel\.dynamic_complexity', 'wide'),
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
        (r'lowlevel\.spectral_energyband_(high|middle_high|middle_low|low)\.\w+', 'deep'),
        (r'lowlevel\.spectral_entropy\.\w+', 'deep'),
        (r'lowlevel\.spectral_flux\.\w+', 'deep'),
        (r'lowlevel\.spectral_(kurtosis|skewness|spread)\.\w+', 'deep'),
        (r'lowlevel\.spectral_rms\.\w+', 'deep'),
        (r'lowlevel\.spectral_rms\.\w+', 'deep'),
        (r'lowlevel\.spectral_rolloff\.\w+', 'deep'),
        (r'lowlevel\.spectral_strongpeak\.\w+', 'deep'),
        (r'lowlevel\.zerocrossingrate\.\w+', 'deep'),
    ]
