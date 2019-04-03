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
    return r'rhythm\.(bpm|beats_count|onset_rate|danceability)'


def hl_tonal_regex():
    return r'tonal\.(chords_changes_rate|chords_number_rate|tuning_frequency)'
